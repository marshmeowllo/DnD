import streamlit as st
import time

# import src.models.model_loader as model_loader
import src.utils.mock as mock
from src.utils.feedback_utils import save_feedback, log_edit_response
from src.components.sidebar import render_sidebar

CHAT_STREAM_DELAY = 0.005

# Initialize session state
for key in ["history_vanilla", "history_trained"]:
    if key not in st.session_state:
        st.session_state[key] = []
if "last_vote_submitted" not in st.session_state:
    st.session_state["last_vote_submitted"] = True

st.header('Dungeons and Dragons', divider="gray")

def chat_stream(user_input, model_name):
    # response = model_loader.generate_response_with_role(temperature, top_p, top_k, model_name=model_name, user_input=user_input)
    response = mock.mock_generate_response(user_input, model_name, temperature, top_p, top_k)

    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)

def handle_model_history(model_name, user_msg, assistant_msg, index):
    st.markdown("**Model A**" if model_name=='vanilla' else "**Model B**")
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

def show_vote_ui(user_prompt, vanilla_res, trained_res):
    vote = st.radio("Which response do you prefer?", ["Model A", "Model B"])
    if vote and st.button("Submit Vote"):
        save_feedback(user_prompt, vanilla_res, trained_res, vote)
        st.success(f"Vote for {vote} recorded")
        st.session_state['last_vote_submitted'] = True

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
                handle_model_history("vanilla", message, vanilla_msg, i + 1)
            with col2:
                handle_model_history("trained", message, trained_msg, i + 1)

            if i + 1 == n - 1 and not st.session_state["last_vote_submitted"]:
                show_vote_ui(message['content'], vanilla_msg['content'], trained_msg['content'])

            continue

temperature, top_p, top_k = render_sidebar(st.session_state)

if not st.session_state['last_vote_submitted']:
    st.warning("Please vote on the last response before continuing")
    st.chat_input("Say something", disabled=True)
elif prompt := st.chat_input("Say something"):
    st.session_state['last_vote_submitted'] = False

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history_vanilla.append({"role": "user", "content": prompt})
    st.session_state.history_trained.append({"role": "user", "content": prompt})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model A**")
        with st.chat_message("assistant"):
            res1 = st.write_stream(chat_stream(st.session_state.history_vanilla, "Vanilla"))
        st.session_state.history_vanilla.append({"role": "assistant", "content": "".join(res1)})
    with col2:
        st.markdown("**Model B**")
        with st.chat_message("assistant"):
            res2 = st.write_stream(chat_stream(st.session_state.history_trained, "Trained"))
        st.session_state.history_trained.append({"role": "assistant", "content": "".join(res2)})

    st.rerun()

print('------------------------------------------------')
print(st.session_state)
print('------------------------------------------------')