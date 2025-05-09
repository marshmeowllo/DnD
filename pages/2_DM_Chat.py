import streamlit as st
import time

from config import MODEL_MAP
from src.components.sidebar import render_sidebar
from src.models.model import generate_response
from src.utils.chat import chat_stream
from src.utils.model_utils import init_models

st.header('Dungeons and Dragons', divider="gray")

if 'players' not in st.session_state or 'tool_calling' not in st.session_state:
    st.error("Session state not initialized (players or tool calling). Please go back to the home page.")
    st.stop()

st.sidebar.title("D&D Dungeon Master")
cur_player = st.sidebar.selectbox("Player name", st.session_state['players'])
model_choice = st.sidebar.selectbox("Choose a model", ["Model A", "Model B", "Model C", "Model D"])
temperature, top_p, top_k = render_sidebar(st.session_state)

if 'model_choice' not in st.session_state or st.session_state['model_choice'] != model_choice:
    st.session_state['model_choice'] = model_choice
    init_models(model_choice)

if len(st.session_state['players']) == 0:
    st.warning("No player available. Create one first.")
    st.chat_input("Say something", disabled=True)
else:
    # Chat History
    for msg in st.session_state['history']:
        with st.chat_message(msg["role"]):
            st.write(msg['content'])

    if prompt := st.chat_input("Say something"):
        user_msg = f"{cur_player}: {prompt}" 
        st.session_state['history'].append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("assistant"):
            response = generate_response(cur_player, prompt, temperature, top_p, top_k, MODEL_MAP[model_choice])
            res = st.write_stream(chat_stream(response))
                
        st.session_state['history'].append({"role": "assistant", "content": "".join(res)})
