import streamlit as st

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

st.title("Welcome to the D&D Dungeon Master App")   
st.markdown("Use the sidebar to create characters or start the game session with the DM.")

# print('------------------------------------------------')
# print(st.session_state)
# print('------------------------------------------------')