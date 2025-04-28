import streamlit as st
import time
import json

# import model_lodder
import mock

CHAT_STREAM_DELAY = 0.005

st.header('Dungeons and Dragons', divider="gray")

def chat_stream(user_input, model_name):
    # response = model_lodder.generate_response_with_role(temperature, top_p, top_k, model_name=model_name, user_input=user_input)
    response = mock.mock_generate_response(user_input, model_name, temperature, top_p, top_k)

    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)

def save_feedback(index, is_trained):
    if is_trained:
        st.session_state.history_trained[index]["feedback"] = st.session_state[f"feedback_trained_{index}"]
    else:
        st.session_state.history_vanilla[index]["feedback"] = st.session_state[f"feedback_vanilla_{index}"]

def log_edit_response(prompt, original, edited, model):
    out = {
        "prompt": prompt,
        "original": original,
        "edited": edited,
        "model": model
    }
    with open("chat_edit_response.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")

def handle_model_history(model_name, user_msg, assistant_msg, index, is_trained):
    st.markdown(f"**{model_name}**")
    with st.chat_message("assistant"):
        edit_key = f"edit_enable_{model_name}_{index}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = False
        
        if not st.session_state[edit_key]:
            st.write(assistant_msg["content"])
            if st.button("Edit", key=f"enable_edit_{model_name}_{index}"):
                st.session_state[edit_key] = True
                st.rerun()
        else:
            edited = st.text_area("Edit Response", value=assistant_msg["content"], key=f"edit_{model_name}_{index}")
            if st.button("Save", key=f"save_{model_name}_{index}", on_click=log_edit_response, args=[user_msg["content"], assistant_msg["content"], edited, model_name]):
                st.session_state[f"history_{model_name}"][index]["content"] = edited
                st.session_state[edit_key] = False
                st.rerun()

        feedback_key = f"feedback_{model_name}_{index}"
        if feedback_key not in st.session_state:
            feedback = assistant_msg.get("feedback", None)
            st.session_state[feedback_key] = feedback
        st.feedback(
            "thumbs",
            key=feedback_key,
            disabled=st.session_state[feedback_key] is not None,
            on_change=save_feedback,
            args=[index, is_trained],
        )

def render_sidebar():
    st.sidebar.header('Settings')
    history = {
        "vanilla": st.session_state.history_vanilla,
        "trained": st.session_state.history_trained,
    }
    st.sidebar.download_button(
        label="Download Conversation", 
        data=json.dumps(history, indent=2),
        file_name="conversation.json",
        mime="application/json",
        icon=":material/download:"
    )
    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)
    top_p = st.sidebar.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9)
    top_k = st.sidebar.slider("Top-k", min_value=1, max_value=100, value=50)
    return temperature, top_p, top_k

for key in ["history_vanilla", "history_trained"]:
    if key not in st.session_state:
        st.session_state[key] = []

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
                handle_model_history("vanilla", message, vanilla_msg, i + 1, False)
            with col2:
                handle_model_history("trained", message, trained_msg, i + 1, True)

            continue

temperature, top_p, top_k = render_sidebar()

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history_vanilla.append({"role": "user", "content": prompt})
    st.session_state.history_trained.append({"role": "user", "content": prompt})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vanilla Model**")
        with st.chat_message("assistant"):
            res1 = st.write_stream(chat_stream(st.session_state.history_vanilla, "Vanilla"))
        st.session_state.history_vanilla.append({"role": "assistant", "content": "".join(res1)})
    with col2:
        st.markdown("**Trained Model**")
        with st.chat_message("assistant"):
            res2 = st.write_stream(chat_stream(st.session_state.history_trained, "Trained"))
        st.session_state.history_trained.append({"role": "assistant", "content": "".join(res2)})

    st.rerun()

print('------------------------------------------------')
print(st.session_state)
print('------------------------------------------------')