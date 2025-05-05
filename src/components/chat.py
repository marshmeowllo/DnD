import streamlit as st

from src.utils.feedback_utils import save_feedback, log_edit_response

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

def _finalize_vote(chosen_res):
    if len(st.session_state['history']) >= 2 and st.session_state['history'][-1]['role'] == 'assistant' and st.session_state['history'][-1]['role'] == 'assistant':
        st.session_state['history'] = st.session_state['history'][:-2]
    st.session_state['history'].append({"role": "assistant", "content": chosen_res})
    st.session_state['last_vote_submitted'] = True
    st.session_state['last_interaction'] = None
    st.rerun()

def show_vote_ui(user_prompt, vanilla_res, trained_res):
    vote = st.radio("Which response do you prefer?", ["Model A", "Model B"])
    if vote and st.button("Submit Vote"):
        save_feedback(user_prompt, vanilla_res, trained_res, vote)
        if vote == 'Model A':
            _finalize_vote(vanilla_res)
        else:
            _finalize_vote(trained_res)