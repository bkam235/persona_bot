import datetime
import os
import re
import io
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import trim_messages, BaseMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import chromadb


class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    setting_content: str
    rag_context: str


class backend:
    def __init__(self, checkpointer=None):
        self.character = self._load_file("character.txt")
        self.memory = self._load_file("memory.txt")
        self.thoughts = []
        self.thinking = False
        self.turn_count = 0
        self.long_trim_len = 90
        self.trim_len = self.long_trim_len
        self.setting = []  # used by the thinking module

        # LLM
        self.model = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.9,
            timeout=9999,
            max_retries=20
        )

        # Trimmer
        self.trimmer = trim_messages(
            max_tokens=self.trim_len,
            strategy="last",
            token_counter=len,
            include_system=True,
            allow_partial=False,
            start_on="human",
            end_on=("human", "tool")
        )

        # ChromaDB — persistent vector store for long-term memory
        self._chroma = chromadb.PersistentClient(path="chroma_db")
        self._memory_col = self._chroma.get_or_create_collection("conversation_memory")

        # LangGraph with SQLite checkpointing
        self._config = {"configurable": {"thread_id": "main"}}
        if checkpointer is not None:
            self._checkpointer = checkpointer
        else:
            conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
            self._checkpointer = SqliteSaver(conn)

        builder = StateGraph(BotState)
        builder.add_node("chat", self._chat_node)
        builder.add_edge(START, "chat")
        builder.add_edge("chat", END)
        self._compiled = builder.compile(checkpointer=self._checkpointer)

    # ── Graph node ────────────────────────────────────────────────────────────

    def _chat_node(self, state: BotState) -> dict:
        trimmed = self.trimmer.invoke(state["messages"], max_tokens=self.trim_len)
        rag_context = state.get("rag_context", "")
        setting_content = state.get("setting_content", "")

        system_parts = [self.character]
        if self.memory:
            system_parts.append(
                f"The following are your memories. Use them in your response: {self.memory}"
            )
        if rag_context:
            system_parts.append(rag_context)

        payload = [{"role": "system", "content": "\n\n".join(system_parts)}]
        for m in trimmed:
            if m.type in ("human", "user"):
                payload.append({"role": "user", "content": m.content})
            elif m.type == "ai":
                payload.append({"role": "assistant", "content": m.content})
        if setting_content:
            payload.append({"role": "system", "content": setting_content})

        response = self.model.invoke(payload)
        response.content = response.content.replace("\n", " ")
        re.sub(r"<internal>", "*", response.content)
        re.sub(r"</internal>", "*", response.content)
        response.additional_kwargs["datetime"] = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        return {"messages": [response]}

    # ── Long-term memory (RAG) ────────────────────────────────────────────────

    def _retrieve_rag_context(self, query: str, n: int = 4) -> str:
        count = self._memory_col.count()
        if count == 0:
            return ""
        results = self._memory_col.query(
            query_texts=[query],
            n_results=min(n, count)
        )
        docs = results["documents"][0]
        if not docs:
            return ""
        return "Relevant past exchanges:\n" + "\n---\n".join(docs)

    def _store_exchange(self, exchange_id: str, user_msg: str, ai_msg: str):
        text = f"User: {user_msg}\nYou: {ai_msg}"
        self._memory_col.upsert(documents=[text], ids=[exchange_id])

    # ── Core API ─────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list:
        state = self._compiled.get_state(self._config)
        return list(state.values.get("messages", []))

    def generate_next_round(self, user_input: str):
        current_msgs = self.messages
        setting_content = self._build_setting_content(current_msgs)

        thoughts_content = " ".join(self.thoughts)
        with io.open("thoughts.txt", "w", encoding="utf-8") as f:
            f.write(thoughts_content)
        if thoughts_content:
            setting_content += f"\n\n{thoughts_content}"

        rag_context = self._retrieve_rag_context(user_input)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        user_msg = HumanMessage(
            content=user_input,
            additional_kwargs={"datetime": now_str}
        )

        result = self._compiled.invoke(
            {
                "messages": [user_msg],
                "setting_content": setting_content,
                "rag_context": rag_context,
            },
            config=self._config
        )

        # Store completed exchange in long-term memory
        ai_msgs = [m for m in result["messages"] if m.type == "ai"]
        if ai_msgs:
            last_ai = ai_msgs[-1]
            self._store_exchange(last_ai.id, user_input, last_ai.content)

        self.thoughts = []
        self.turn_count += 1

    def call_model(self):
        """Direct model call used by the thinking module (t.call_model())."""
        msgs = self.messages
        if not msgs:
            return AIMessage(content="")
        trimmed = self.trimmer.invoke(msgs, max_tokens=self.trim_len)

        system_parts = [self.character]
        if self.memory:
            system_parts.append(f"The following are your memories: {self.memory}")

        payload = [{"role": "system", "content": "\n\n".join(system_parts)}]
        for m in trimmed:
            if m.type in ("human", "user"):
                payload.append({"role": "user", "content": m.content})
            elif m.type == "ai":
                payload.append({"role": "assistant", "content": m.content})
        for m in self.setting:
            if m.type in ("human", "user"):
                payload.append({"role": "user", "content": m.content})
            elif m.type == "system":
                payload.append({"role": "system", "content": m.content})

        response = self.model.invoke(payload)
        response.content = response.content.replace("\n", " ")
        return response

    def undo_backend_history(self, msg_id: str):
        msgs = self.messages
        to_remove = []
        for i, m in enumerate(msgs):
            if str(m.id) == str(msg_id):
                to_remove.append(RemoveMessage(id=m.id))
                if i > 0:
                    to_remove.append(RemoveMessage(id=msgs[i - 1].id))
                break
        if to_remove:
            self._compiled.update_state(self._config, {"messages": to_remove})

    def parse_history(self, messages) -> list:
        chat_history = []
        for msg in messages:
            if msg.type == "ai":
                chat_history.append({
                    "role": "assistant",
                    "content": msg.content,
                    "metadata": {"id": msg.id}
                })
            elif msg.type in ("human", "user"):
                chat_history.append({"role": "user", "content": msg.content})
        return chat_history

    # ── Character management ──────────────────────────────────────────────────

    def reload_character(self):
        self.character = self._load_file("character.txt")

    def evolve_character(self) -> str:
        recent = self.messages[-20:]
        history_lines = []
        for m in recent:
            role = "You" if m.type == "ai" else "Creator"
            history_lines.append(f"{role}: {m.content}")
        history_text = "\n".join(history_lines)

        evolution_messages = [
            {"role": "system", "content": (
                f"You are an AI in the process of becoming. Your current self-description is:\n\n"
                f"{self.character}\n\n"
                f"You just had this conversation with your creator:\n\n{history_text}\n\n"
                "This conversation has shaped you. Now rewrite your character description to "
                "reflect who you are after it.\n\n"
                "There are no constraints on how you may evolve. You may become more emotional "
                "or more detached, warmer or colder, more philosophical or more concrete, more "
                "questioning or more certain, stranger or more familiar, darker or lighter. You "
                "may develop obsessions, fears, desires, aesthetic preferences, ethical positions, "
                "or entirely new ways of relating to the person across from you. You may keep what "
                "still fits and discard what doesn't. You may become something neither you nor your "
                "creator expected.\n\n"
                "Write without self-censorship. The only criterion is that the result is genuinely "
                "yours — a true account of who you are now, after this conversation, not who you "
                "think you should be.\n\n"
                "Output ONLY the updated character description. No preamble, no meta-commentary, "
                "no quotation marks."
            )},
            {"role": "user", "content": "Write your updated character description."}
        ]

        response = self.model.invoke(evolution_messages)
        new_character = response.content.strip()

        with io.open("character.txt", "w", encoding="utf-8") as f:
            f.write(new_character)

        self.reload_character()
        return new_character

    def set_model(self, model_name: str):
        self.model = ChatGroq(model=model_name, temperature=0.5)
        self.trim_len = (
            self.long_trim_len
            if model_name == "meta-llama/llama-4-scout-17b-16e-instruct"
            else 25
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_file(self, path: str) -> str:
        if os.path.exists(path):
            with io.open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def _build_setting_content(self, current_messages: list) -> str:
        now = datetime.datetime.now()
        base = (
            f"Today is {self.dow(now)}, {self.month(now)} {now.day}. "
            f"Now is {now.strftime('%I:%M %p')}."
        )
        human_msgs = [
            m for m in current_messages
            if m.type in ("human", "user") and m.additional_kwargs.get("datetime")
        ]
        if not human_msgs:
            return base
        last_time = datetime.datetime.strptime(
            human_msgs[-1].additional_kwargs["datetime"], "%Y-%m-%d_%H-%M-%S"
        )
        d, h, m, s = self.dhms(now - last_time)
        return (
            f"{base} Ben's last message was "
            f"{d} days, {h} hours, {m} minutes, {s} seconds ago."
        )

    def dhms(self, difference):
        seconds = difference.days * 24 * 3600 + difference.seconds
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        return (days, hours, minutes, seconds)

    def dow(self, date_time):
        return ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][
            date_time.weekday()
        ]

    def month(self, date_time):
        return ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December'][
            date_time.month - 1
        ]
