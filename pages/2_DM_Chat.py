import streamlit as st
from src.components.chat import show_vote_ui
from src.components.sidebar import render_sidebar
import src.models.model_loader_new as model_loader
from config import CHAT_STREAM_DELAY
import time

st.header('Dungeons and Dragons', divider="gray")

n = len(st.session_state['history'])
for i, msg in enumerate(st.session_state['history']):
    if not st.session_state['last_vote_submitted'] and i == n -2:
        break
    with st.chat_message(msg['role']):
        st.write(msg['content'])

st.sidebar.title("D&D Dungeon Master")
cur_player = st.sidebar.selectbox("Player name", st.session_state['players'])
temperature, top_p, top_k = render_sidebar(st.session_state)

def chat_stream(user_input, model_name, temperature, top_k, top_p):
    response = model_loader.generate_response(cur_player, user_input[-1]['content'], temperature, top_p, top_k, model_name)

    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)

if not st.session_state['last_vote_submitted'] and st.session_state['last_interaction']:
    prompt, res_a, res_b = st.session_state['last_interaction']
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model A**")
        with st.chat_message("assistant"):
            st.write(res_a)
        
    with col2:
        st.markdown("**Model B**")
        with st.chat_message("assistant"):
            st.write(res_b)
    show_vote_ui(prompt, res_a, res_b)
    st.chat_input("Say something", disabled=True)
elif st.session_state['last_vote_submitted']:
    if prompt := st.chat_input("Say something"):
        st.session_state['last_vote_submitted'] = False
        prompt = f"{cur_player}: " + prompt

        with st.chat_message("user"):
            st.write(prompt)
        st.session_state['history'].append({"role": "user", "content": prompt})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model A**")
            with st.chat_message("assistant"):
                res_a = st.write_stream(chat_stream(st.session_state['history'], "vanilla", temperature, top_k, top_p))
            
        with col2:
            st.markdown("**Model B**")
            with st.chat_message("assistant"):
                res_b = st.write_stream(chat_stream(st.session_state['history'], "trained", temperature, top_k, top_p))
        
        st.session_state['last_interaction'] = (prompt, res_a, res_b)
        
        st.session_state['history'].append({"role": "assistant", "content": "".join(res_a)})
        st.session_state['history'].append({"role": "assistant", "content": "".join(res_b)})

        st.rerun()
else:
    st.warning("Please vote on the last response before continuing")
    st.chat_input("Say something", disabled=True)