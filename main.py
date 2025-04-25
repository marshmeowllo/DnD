import streamlit as st
import time
import json

import model_lodder

st.header('Dungeons and Dragons', divider="gray")

def chat_stream(user_input):
    response = model_lodder.generate_response_with_role(role="player1", user_input=user_input)

    for char in response:
        yield char
        time.sleep(0.005)

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        user_input = prompt
        response = st.write_stream(chat_stream(user_input))
        
        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )
    st.session_state.history.append({"role": "assistant", "content": response})

# print('------------------------------------------------')
# print(st.session_state)
# print('------------------------------------------------')

st.sidebar.header('Settings')

with st.sidebar:
    st.download_button(
        label="Download Conversation", 
        data=json.dumps(st.session_state.history, indent=2),
        file_name="conversation.json",
        mime="application/json",
        icon=":material/download:"
    )
