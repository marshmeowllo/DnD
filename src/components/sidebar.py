import json
import streamlit as st

def render_sidebar(session_state):
    st.sidebar.header('Settings')
    history = {
        "vanilla": session_state.history_vanilla,
        "trained": session_state.history_trained,
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