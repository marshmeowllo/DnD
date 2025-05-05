import streamlit as st
import time
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Mock
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# import src.models.model_loader as model_loader
from src.components.chat import handle_model_history, show_vote_ui
import src.utils.mock as mock
from src.components.sidebar import render_sidebar

CHAT_STREAM_DELAY = 0.005

# Initialize session state
for key in ["history_vanilla", "history_trained"]:
    if key not in st.session_state:
        st.session_state[key] = []
if "last_vote_submitted" not in st.session_state:
    st.session_state["last_vote_submitted"] = True
if "vectorstore" not in st.session_state:
    character_docs = [Document(page_content="Name: Throg\nRace: Orc\nClass:Barbarian\nLevel: 1\n", metadata={"player": "jump", "type":"character_sheet", "name": "Throg"})]
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-modernbert-base",model_kwargs={'trust_remote_code': True})
    st.session_state['vectorstore'] = FAISS.from_documents(character_docs, embedding)

retriever = st.session_state['vectorstore'].as_retriever()

temp_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Dungeon Master AI. Use the following character information to help answer the player's question.

Character Info:
{context}

Question: {question}
Answer:"""
)

temp_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_keys=os.getenv('GOOGLE_API_KEY'),  temperature=0.7)
temp_qa_chain = RetrievalQA.from_chain_type(
    llm=temp_llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": temp_prompt}
)

st.header('Dungeons and Dragons', divider="gray")

def chat_stream(user_input, model_name):
    # response = model_loader.generate_response_with_role(temperature, top_p, top_k, model_name=model_name, user_input=user_input)
    if model_name == 'vanilla':
        response = temp_qa_chain.run(user_input[0]['content'])
    else:
        response = temp_qa_chain.run(user_input[0]['content'])

    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)

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
            res1 = st.write_stream(chat_stream(st.session_state.history_vanilla, "vanilla"))
        st.session_state.history_vanilla.append({"role": "assistant", "content": "".join(res1)})
    with col2:
        st.markdown("**Model B**")
        with st.chat_message("assistant"):
            res2 = st.write_stream(chat_stream(st.session_state.history_trained, "trained"))
        st.session_state.history_trained.append({"role": "assistant", "content": "".join(res2)})

    st.rerun()

print('------------------------------------------------')
print(st.session_state)
print('------------------------------------------------')