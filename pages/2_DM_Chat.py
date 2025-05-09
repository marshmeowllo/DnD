import streamlit as st
from src.components.sidebar import render_sidebar
from config import CHAT_STREAM_DELAY
import time

from src.models.model import generate_response
from src.models.model import LlamaChat, ToolCalling
from src.tools.tools import spell_retrieve, user
from src.utils.graph_builder import change_model

MODEL_MAP = {"Model A": "gemini-2.0-flash-001", "Model B": "gemini-1.5-flash", "Model C": "gemini-1.5-flash-8b"}

st.header('Dungeons and Dragons', divider="gray")

st.sidebar.title("D&D Dungeon Master")
cur_player = st.sidebar.selectbox("Player name", st.session_state['players'])
model_choice = st.sidebar.selectbox("Choose a model", ["Model A", "Model B", "Model C"], on_change=change_model)
temperature, top_p, top_k = render_sidebar(st.session_state)

st.session_state['tool_calling'] = ToolCalling(model_name=MODEL_MAP[model_choice], tools=[spell_retrieve, user])
st.session_state['llama'] = LlamaChat(model_name=MODEL_MAP[model_choice])
    
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
