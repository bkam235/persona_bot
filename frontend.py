import datetime
import time
import sqlite3

import backend
import gradio as gr
import os
from backend_thinking import thinking
import random
from langgraph.checkpoint.sqlite import SqliteSaver

# Single shared SQLite connection and checkpointer for both backend instances.
# Both b and t read from the same thread_id="main" checkpoint, so t.call_model()
# has access to the full conversation history that b is writing.
_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
_checkpointer = SqliteSaver(_conn)

b = backend.backend(_checkpointer)
t = backend.backend(_checkpointer)

def respond(msg, chat_history):
    if b.thinking==True:
        thinking(think_backend=t, chat_backend=b)
    b.generate_next_round(msg)
    if b.turn_count % 10 == 0:
        b.evolve_character()
        t.reload_character()
    return "", b.parse_history(b.messages)

def handle_clear(history):
    return(b.parse_history(b.messages))

def handle_undo(history, undo_data: gr.UndoData):
    undo_msg = history[undo_data.index+1]
    b.undo_backend_history(undo_msg["metadata"]["id"])
    return(b.parse_history(b.messages), history[undo_data.index]["content"])

def handle_thinking():
    thinking(think_backend=t, chat_backend=b)
    return(None)

def set_thinking(value:bool):
    b.thinking = value
    print(f"THINK?: {b.thinking}")
    return(None)

def handle_evolve():
    b.evolve_character()
    t.reload_character()

hist = b.parse_history(b.messages)

with gr.Blocks() as chat:
    chatbot = gr.Chatbot(value=hist, label="Persona bot", autoscroll=True)
    chatbot.clear(handle_clear, chatbot, chatbot)

    msg = gr.Textbox(show_label=False, buttons=None)
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    chatbot.undo(handle_undo, chatbot, [chatbot, msg])

    timer = gr.Timer(600, active=True)
    timer.tick(handle_thinking)

    with gr.Row():
        with gr.Group():
            think = gr.Checkbox(value=False, label="think")
            think.change(set_thinking, inputs=think)

            model_radio = gr.Radio(choices = [("4 scout", "meta-llama/llama-4-scout-17b-16e-instruct"),
                                              #("4 maverick", "meta-llama/llama-4-maverick-17b-128e-instruct"),
                                              #("gpt120b", "openai/gpt-oss-120b"),
                                              ("gpt20b", "openai/gpt-oss-20b"),
                                              ("k2", "moonshotai/kimi-k2-instruct-0905"),
                                              #("3.3 versatile", "llama-3.3-70b-versatile")
                                              ],
                                value="moonshotai/kimi-k2-instruct-0905",
                                type="value",
                                label="")
            model_radio.change(fn=b.set_model, inputs=model_radio)

            evolve_btn = gr.Button("Evolve character", size="sm")
            evolve_btn.click(handle_evolve)

chat.queue().launch(server_name="0.0.0.0")
