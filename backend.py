import datetime
import os
import re
import io
from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

class backend:
    def __init__(self):
        self.initial_scenario = "This is your first chat with your user. You're waiting for the first message."

        if os.path.exists("character.txt"):
            with io.open("character.txt", "r", encoding="utf-8") as f:
                self.character = f.read()
        else:
            self.character = ""

        if os.path.exists("memory.txt"):
            with io.open("memory.txt", "r", encoding="utf-8") as f:
                self.memory = f.read()
        else:
            self.memory = ""

        self.setting = [BaseMessage(type="tool",
                                    content="",
                                    tool_call_id="datetime")]

        # Initialize history
        self.history = SQLChatMessageHistory(
            session_id="ava", connection="sqlite:///bot.db"
        )
        if len(self.history.messages) == 0:
            self.history.add_user_message(self.initial_scenario)

        # Initialize model
        self.model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.9, timeout=9999, max_retries=20)

        # Trimmer
        self.long_trim_len = 90
        self.trim_len = self.long_trim_len
        self.trimmer = trim_messages(
            max_tokens=self.trim_len,
            strategy="last",
            token_counter=len,
            include_system=True,
            allow_partial=False,
            start_on="human",
            end_on=("human", "tool")
        )

        # Chat prompt template
        self.prompt_template_memory = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f" {self.character} The following are your memories. Use them in your response: {self.memory}"
                ),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="setting")
            ]
        )

        self.prompt_template = self.prompt_template_memory
        self.thoughts = []
        self.thinking = False

    def set_model(self, model_name):
        self.model = ChatGroq(model=model_name, temperature=0.5)
        if (model_name == "meta-llama/llama-4-scout-17b-16e-instruct"):
            self.trim_len = self.long_trim_len
            self.prompt_template = self.prompt_template_memory
        else:
            self.trim_len = 25

    def call_model(self):
        trimmed_messages = self.trimmer.invoke(self.history.messages, max_tokens=self.trim_len)
        prompt = self.prompt_template.invoke(input={"messages": trimmed_messages,
                                                    "setting": self.setting})
        prompt = self.parse_history(prompt)
        #print(prompt[-5:])
        response = self.model.invoke(prompt)

        response.content = response.content.replace('\n', ' ')

        re.sub(r"<internal>", "*", response.content)
        re.sub(r"</internal>", "*", response.content)
        return response

    def dhms(self, difference):
        seconds = difference.days * 24 * 3600 + difference.seconds
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        return (days, hours, minutes, seconds)

    def dow(self, date_time):
        i = date_time.weekday()
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return weekdays[i]

    def month(self, date_time):
        i = date_time.month
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        return months[i-1]

    def generate_time_message(self):
        types = [msg.type for msg in self.history.messages]
        human_msgs = [type == "human" for type in reversed(types)]
        last_human_msg = [i+1 for i, x in enumerate(human_msgs) if x][1] * -1

        last_time = self.history.messages[last_human_msg].additional_kwargs["datetime"]
        last = datetime.datetime.strptime(last_time, "%Y-%m-%d_%H-%M-%S")
        now = datetime.datetime.now()
        difference = self.dhms(now - last)
        msg = (f"""Today is {self.dow(now)}, {self.month(now)} {now.day}. Now is {now.strftime("%I:%M %p")}. Ben's last message was {difference[0]} days, {difference[1]} hours, {difference[2]} minutes, {difference[3]} seconds ago.""")
        return msg
    def generate_next_round(self, user_input):
        self.history.add_message(BaseMessage(type="human",
                                             content=user_input,
                                             additional_kwargs={"datetime": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}))
        date_msg = self.generate_time_message()
        self.setting = []
        self.setting.append(BaseMessage(type="tool",
                                             content=date_msg,
                                             tool_call_id="datetime"))

        thoughts_content = " ".join(self.thoughts)

        with io.open("thoughts.txt", "w", encoding="utf-8") as f:
            f.write(thoughts_content)

        self.setting.append(BaseMessage(type="tool",
                                content=f"""{thoughts_content}""",
                                tool_call_id=""))

        reply = self.call_model()
        self.thoughts = []
        reply.additional_kwargs = {"datetime": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}
        self.history.add_message(reply)

    def undo_backend_history(self, msg_id):
        history_ids = [msg.id for msg in self.history.messages]
        backend_index = history_ids.index(msg_id)-1
        new_backend_history = self.history.messages[:backend_index]
        self.history.clear()
        self.history.add_messages(new_backend_history)

    def parse_history(self, history):
        chat_history = list()
        for msg in history.messages:
            if msg.type == "ai":
                chat_history.append({"role": "assistant",
                                     "content": msg.content,
                                     "metadata": {"id": msg.id}})
            if msg.type == "tool" and msg.tool_call_id != 'datetime':
                chat_history.append({"role": "system", "content": msg.content})
            if msg.type == "tool" and msg.tool_call_id == 'datetime':
                chat_history.append({"role": "system", "content": msg.content})
            if msg.type == "system":
                chat_history.append({"role": "system", "content": msg.content})
            elif msg.type == "human" or msg.type == "user":
                chat_history.append({"role": "user", "content": msg.content})
            else:
                continue
        return (chat_history)



