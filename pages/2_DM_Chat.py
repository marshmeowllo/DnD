import streamlit as st
import time

from config import CHAT_STREAM_DELAY, MODEL_MAP
from src.components.sidebar import render_sidebar
from src.models.model import generate_response
from src.utils.model_utils import init_models

st.header('Dungeons and Dragons', divider="gray")

if 'players' not in st.session_state:
    st.error("Session state not initialized. Please go back to the home page.")
    st.stop()

st.sidebar.title("D&D Dungeon Master")
cur_player = st.sidebar.selectbox("Player name", st.session_state['players'])
model_choice = st.sidebar.selectbox("Choose a model", ["Model A", "Model B", "Model C"])
temperature, top_p, top_k = render_sidebar(st.session_state)

if 'model_choice' not in st.session_state or st.session_state['model_choice'] != model_choice:
    st.session_state['model_choice'] = model_choice
    init_models(model_choice)
    
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

        st.session_state['history'].append({"role": "user", "content": f"{cur_player}: " + prompt})

        with st.chat_message("assistant"):
            res = st.write_stream(chat_stream(prompt, MODEL_MAP[model_choice], temperature, top_k, top_p))
                
        st.session_state['history'].append({"role": "assistant", "content": "".join(res)})
