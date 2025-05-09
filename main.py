import streamlit as st

from src.models.model import State, chatbot, tool
from src.utils.initialization import init_spellstore, init_vectorstore, load_embeddings, load_players
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "vectorstore" not in st.session_state:
    embedding = load_embeddings()
    st.session_state['vectorstore'] = init_vectorstore(embedding)
# if "spellstore" not in st.session_state:
#     st.session_state['spellstore'] = init_spellstore()
if "players" not in st.session_state:
    st.session_state['players'] = load_players()
if "graph" not in st.session_state:
    st.session_state['memory'] = MemorySaver()
    graph = StateGraph(State)
    graph.add_edge(START, "tool call")
    graph.add_node("tool call", tool)
    graph.add_edge("tool call", "chatbot")
    graph.add_node("chatbot", chatbot)
    graph.add_edge("chatbot", END)
    st.session_state['graph'] = graph.compile(checkpointer=st.session_state['memory'])

st.title("Welcome to the D&D Dungeon Master App")   
st.markdown("Use the sidebar to create characters or start the game session with the DM.")

# print('------------------------------------------------')
# print(st.session_state)
# print('------------------------------------------------')