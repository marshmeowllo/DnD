import streamlit as st

from langgraph.checkpoint.memory import MemorySaver

from src.utils.graph_builder import build_graph
from src.utils.initialization import init_spellstore, init_vectorstore, load_embeddings, load_players

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
    st.session_state['graph'] = build_graph(st.session_state['memory'])

st.title("Welcome to the D&D Dungeon Master App")   
st.markdown("Use the sidebar to create characters or start the game session with the DM.")

# print('------------------------------------------------')
# print(st.session_state)
# print('------------------------------------------------')