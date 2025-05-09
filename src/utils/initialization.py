import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import ChatGoogleGenerativeAI
import os

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def init_vectorstore(_embedding):
    index = faiss.IndexFlatL2(len(_embedding.embed_query("test")))
    return FAISS(embedding_function=_embedding, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

@st.cache_resource
def init_dndstore(_embedding):
    return FAISS.load_local(
        "~/DnD/examples/faiss_dnd_index",
        embeddings=_embedding,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm(model_name):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_keys=os.getenv('GOOGLE_API_KEY'),
        temperature=0.7
    )

def load_players():
    temp = []
    for doc in st.session_state['vectorstore'].docstore._dict.values():
        temp.append(doc.metadata['name'])
    return temp