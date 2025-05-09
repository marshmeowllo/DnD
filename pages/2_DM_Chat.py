import streamlit as st
from src.components.chat import handle_model_history, show_vote_ui
from src.components.sidebar import render_sidebar
from config import CHAT_STREAM_DELAY
import time

from src.models.model import generate_response

MODEL_MAP = {"Model A": "trained", "Model B": "trained", "Model C": "vanilla"}

st.header('Dungeons and Dragons', divider="gray")

n = len(st.session_state['history'])
for i, msg in enumerate(st.session_state['history']):
    if not st.session_state['last_vote_submitted'] and i == n -2:
        break
    with st.chat_message(msg['role']):
        st.write(msg['content'])

st.sidebar.title("D&D Dungeon Master")
cur_player = st.sidebar.selectbox("Player name", st.session_state['players'])
model_choice = st.sidebar.selectbox("Choose a model", ["Model A", "Model B", "Model C"])
temperature, top_p, top_k = render_sidebar(st.session_state)

def chat_stream(user_input, model_name, temperature, top_k, top_p):
    response = generate_response(cur_player, user_input, temperature, top_p, top_k, model_name)

    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)

if len(st.session_state['players']) == 0:
    st.warning("No player available. Create one first.")
    st.chat_input("Say something", disabled=True)
else:
    # Chat History
    for msg in st.session_state['history']:
        with st.chat_message(msg["role"]):
            st.write(msg['content'])

    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.write(f"{cur_player}: " + prompt)
            
        st.session_state['history'].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            res = st.write_stream(chat_stream(prompt, MODEL_MAP[model_choice], temperature, top_k, top_p))
                
        st.session_state['history'].append({"role": "assistant", "content": "".join(res)})
