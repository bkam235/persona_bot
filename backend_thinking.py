from langchain_core.messages import BaseMessage
import random

def thinking(think_backend, chat_backend):
    if len(chat_backend.thoughts) >= 3:
        return(None)

    think_backend.character = """
    [to be filled from character.txt]
    """

    think_backend.setting = []
    think_backend.setting.append(BaseMessage(type="human",
                                 content="""Your next message should be an internal thought process, reviewing past messages, your character and memories.
                                Consider how much time has passed between messages.
                                Also form an intention what you want to do next, what you want to say next.
                                Write your message as a flow of consciousness.
                                """
                                 ))
    preset_model = think_backend.model.model_name
    think_backend.set_model(random.choice([#"meta-llama/llama-4-scout-17b-16e-instruct",
                                            #"openai/gpt-oss-20b",
                                            "moonshotai/kimi-k2-instruct-0905"]))
    reply = think_backend.call_model()

    chat_backend.thoughts = []
    chat_backend.thoughts.append(f"{reply.content}")

    think_backend.set_model(preset_model)

    return(None)