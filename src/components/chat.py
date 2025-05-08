import streamlit as st

from src.utils.feedback_utils import save_feedback, log_edit_response

def handle_model_history(model_name):
    show_name = "**Model A**" if model_name=='vanilla' else "**Model B**"
    prompt, res_a, res_b = st.session_state['last_interaction']
    assistant_msg = res_a if model_name=='vanilla' else res_b
    st.markdown(show_name)
    with st.chat_message("assistant"):
        edit_key = f"edit_enable_{model_name}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = False
        
        if not st.session_state[edit_key]:
            st.write(assistant_msg)
            if st.button("Edit", key=f"enable_edit_{model_name}"):
                st.session_state[edit_key] = True
                st.rerun()
        else:
            edited = st.text_area("Edit Response", value=assistant_msg, key=f"edit_{model_name}")
            if st.button("Save", key=f"save_{model_name}", on_click=log_edit_response, args=[prompt, assistant_msg, edited, model_name]):
                st.session_state['last_interaction'] = (prompt, edited, res_b) if model_name=='vanilla' else (prompt, res_a, edited)
                st.session_state[edit_key] = False
                st.rerun()

def _finalize_vote(chosen_res):
    if len(st.session_state['history']) >= 2 and st.session_state['history'][-1]['role'] == 'assistant' and st.session_state['history'][-1]['role'] == 'assistant':
        st.session_state['history'] = st.session_state['history'][:-2]
    st.session_state['history'].append({"role": "assistant", "content": chosen_res})
    st.session_state['last_vote_submitted'] = True
    st.session_state['last_interaction'] = None
    # st.session_state['memory'].get()
    st.rerun()

def show_vote_ui():
    prompt, res_a, res_b = st.session_state['last_interaction']
    vote = st.radio("Which response do you prefer?", ["Model A", "Model B"])
    if vote and st.button("Submit Vote"):
        save_feedback(prompt, res_a, res_b, vote)
        if vote == 'Model A':
            _finalize_vote(res_a)
        else:
            _finalize_vote(res_b)