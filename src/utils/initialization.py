import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import ChatGoogleGenerativeAI
import os

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-modernbert-base",
        model_kwargs={'trust_remote_code': True}
    )

@st.cache_resource
def init_vectorstore(_embedding):
    index = faiss.IndexFlatL2(len(_embedding.embed_query("test")))
    return FAISS(embedding_function=_embedding, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_keys=os.getenv('GOOGLE_API_KEY'),
        temperature=0.7
    )

def load_players():
    temp = []
    for doc in st.session_state['vectorstore'].docstore._dict.values():
        temp.append(doc.metadata['player'])
    return temp

@st.cache_resource
def init_spellstore():
    embed_model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    return FAISS.load_local(
        "./examples/faiss_spell_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )