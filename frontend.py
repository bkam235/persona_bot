import datetime
import time

import backend
import gradio as gr
import os
from backend_thinking import thinking
import random

b = backend.backend()
t = backend.backend()

def respond(msg, chat_history):
    if b.thinking==True:
        thinking(think_backend=t, chat_backend=b)
    b.generate_next_round(msg)
    return "", b.parse_history(b.history)

def handle_clear(history):
    return(b.parse_history(b.history))

def handle_undo(history, undo_data: gr.UndoData):
    undo_msg = history[undo_data.index+1]
    b.undo_backend_history(undo_msg["metadata"]["id"])
    return(b.parse_history(b.history), history[undo_data.index]["content"])

def handle_thinking():
    thinking(think_backend=t, chat_backend=b)
    return(None)

def set_thinking(value:bool):
    b.thinking = value
    print(f"THINK?: {b.thinking}")
    return(None)

hist = b.parse_history(b.history)

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

chat.queue().launch(server_name="0.0.0.0")

