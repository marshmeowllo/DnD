import streamlit as st
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_vote_submitted" not in st.session_state:
    st.session_state["last_vote_submitted"] = True
if "last_interaction" not in st.session_state:
    st.session_state["last_interaction"] = None
if "vectorstore" not in st.session_state:
    embedding = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-modernbert-base",model_kwargs={'trust_remote_code': True})
    index = faiss.IndexFlatL2(len(embedding.embed_query("test")))
    st.session_state['vectorstore'] = FAISS(embedding_function=embedding, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
if "players" not in st.session_state:
    st.session_state['players'] = []

st.title("Welcome to the D&D Dungeon Master App")   
st.markdown("Use the sidebar to create characters or start the game session with the DM.")

# print('------------------------------------------------')
# print(st.session_state)
# print('------------------------------------------------')