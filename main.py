import streamlit as st
import time
import json

# import model_lodder
import mock

st.header('Dungeons and Dragons', divider="gray")

def chat_stream(user_input):
    # response = model_lodder.generate_response_with_role(temperature, top_p, top_k, user_input=user_input)
    response = mock.mock_generate_response(user_input, temperature, top_p, top_k)

    for char in response:
        yield char
        time.sleep(0.005)

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
        
        n = len(st.session_state.history)
        vanilla_msg = st.session_state.history[i + 1] if i + 1 < n else None
        trained_msg = st.session_state.history[i + 2] if i + 2 < n else None

        if vanilla_msg and trained_msg and vanilla_msg["role"] == "assistant_vanilla" and trained_msg["role"] == "assistant_trained":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Vanilla Model**")
                with st.chat_message("assistant"):
                    st.write(vanilla_msg["content"])

                    feedback = message.get("feedback", None)
                    st.session_state[f"feedback_{i + 1}"] = feedback
                    st.feedback(
                        "thumbs",
                        key=f"feedback_{i + 1}",
                        disabled=feedback is not None,
                        on_change=save_feedback,
                        args=[i + 1],
                    )
            with col2:
                st.markdown("**Trained Model**")
                with st.chat_message("assistant"):
                    st.write(trained_msg["content"])
                    feedback = message.get("feedback", None)
                    st.session_state[f"feedback_{i + 2}"] = feedback
                    st.feedback(
                        "thumbs",
                        key=f"feedback_{i + 2}",
                        disabled=feedback is not None,
                        on_change=save_feedback,
                        args=[i + 2],
                    )

            continue

st.sidebar.header('Settings')

with st.sidebar:
    st.download_button(
        label="Download Conversation", 
        data=json.dumps(st.session_state.history, indent=2),
        file_name="conversation.json",
        mime="application/json",
        icon=":material/download:"
    )

st.sidebar.header("Model Parameters")

temperature = st.sidebar.slider(
    "Temperature", min_value=0.1, max_value=2.0, value=1.0
)

top_p = st.sidebar.slider(
    "Top-p", min_value=0.1, max_value=1.0, value=0.9
)

top_k = st.sidebar.slider(
    "Top-k", min_value=1, max_value=100, value=50
)

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vanilla Model**")
        with st.chat_message("assistant"):
            res1 = st.write_stream(chat_stream(st.session_state.history))
        st.session_state.history.append({"role": "assistant_vanilla", "content": "".join(res1)})
    with col2:
        st.markdown("**Trained Model**")
        with st.chat_message("assistant"):
            res2 = st.write_stream(chat_stream(st.session_state.history))
        st.session_state.history.append({"role": "assistant_trained", "content": "".join(res2)})

    st.rerun()

print('------------------------------------------------')
print(st.session_state)
print('------------------------------------------------')