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

def save_feedback(index, is_trained):
    if is_trained:
        st.session_state.history_trained[index]["feedback"] = st.session_state[f"feedback_trained_{index}"]
    else:
        st.session_state.history_vanilla[index]["feedback"] = st.session_state[f"feedback_vanilla_{index}"]

if "history_vanilla" not in st.session_state:
    st.session_state.history_vanilla = []
if "history_trained" not in st.session_state:
    st.session_state.history_trained = []

n = len(st.session_state.history_vanilla)
for i in range(n):
    message = st.session_state.history_vanilla[i]
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
        
        vanilla_msg = st.session_state.history_vanilla[i + 1] if i + 1 < n else None
        trained_msg = st.session_state.history_trained[i + 1] if i + 1 < n else None

        if vanilla_msg and trained_msg and vanilla_msg["role"] == "assistant" and trained_msg["role"] == "assistant":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Vanilla Model**")
                with st.chat_message("assistant"):
                    st.write(vanilla_msg["content"])
                    feedback_key = f"feedback_vanilla_{i}"
                    if feedback_key not in st.session_state:
                        feedback = vanilla_msg.get("feedback", None)
                        st.session_state[feedback_key] = feedback
                    st.feedback(
                        "thumbs",
                        key=feedback_key,
                        disabled=st.session_state[feedback_key] is not None,
                        on_change=save_feedback,
                        args=[i, False],
                    )
            with col2:
                st.markdown("**Trained Model**")
                with st.chat_message("assistant"):
                    st.write(trained_msg["content"])
                    feedback_key = f"feedback_trained_{i}"
                    if feedback_key not in st.session_state:
                        feedback = trained_msg.get("feedback", None)
                        st.session_state[feedback_key] = feedback
                    st.feedback(
                        "thumbs",
                        key=feedback_key,
                        disabled=st.session_state[feedback_key] is not None,
                        on_change=save_feedback,
                        args=[i, True],
                    )

            continue

st.sidebar.header('Settings')

with st.sidebar:
    history = {
        "vanilla": st.session_state.history_vanilla,
        "trained": st.session_state.history_trained,
    }

    st.download_button(
        label="Download Conversation", 
        data=json.dumps(history, indent=2),
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
    st.session_state.history_vanilla.append({"role": "user", "content": prompt})
    st.session_state.history_trained.append({"role": "user", "content": prompt})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vanilla Model**")
        with st.chat_message("assistant"):
            res1 = st.write_stream(chat_stream(st.session_state.history_vanilla))
        st.session_state.history_vanilla.append({"role": "assistant", "content": "".join(res1)})
    with col2:
        st.markdown("**Trained Model**")
        with st.chat_message("assistant"):
            res2 = st.write_stream(chat_stream(st.session_state.history_trained))
        st.session_state.history_trained.append({"role": "assistant", "content": "".join(res2)})

    st.rerun()

print('------------------------------------------------')
print(st.session_state)
print('------------------------------------------------')