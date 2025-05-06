import streamlit as st
import time
import os
import faiss
import random
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Mock
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import src.models.model_loader_new as model_loader
from config import CHAT_STREAM_DELAY, CHARACTER_BACKGROUND, CHARACTER_CLASSES, CHARACTER_RACES
from src.components.chat import handle_model_history, show_vote_ui
from src.components.sidebar import render_sidebar

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_vote_submitted" not in st.session_state:
    st.session_state["last_vote_submitted"] = True
if "last_interaction" not in st.session_state:
    st.session_state["last_interaction"] = None
if "vectorstore" not in st.session_state:
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-modernbert-base",model_kwargs={'trust_remote_code': True})
    index = faiss.IndexFlatL2(len(embedding.embed_query("test")))
    st.session_state['vectorstore'] = FAISS(embedding_function=embedding, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
if "players" not in st.session_state:
    st.session_state['players'] = []

st.sidebar.title("D&D Dungeon Master")
page = st.sidebar.radio("Go to", ["Character Creator", "DM Chat"])
cur_player = st.sidebar.selectbox("Player name", st.session_state['players'])

def chat_stream(user_input, model_name, temperature, top_k, top_p):
    # temp_prompt = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template="""
    # You are a Dungeon Master AI. Use the following character information to help answer the player's question.

    # Character Info:
    # {context}

    # Question: {question}
    # Answer:"""
    # )
    # retriever = st.session_state['vectorstore'].as_retriever()
    # temp_qa_chain = RetrievalQA.from_chain_type(
    #     llm=temp_llm,
    #     retriever=retriever,
    #     chain_type_kwargs={"prompt": temp_prompt}
    # )

    response = model_loader.generate_response(cur_player, user_input[-1]['content'], temperature, top_p, top_k, model_name)
    # Temporarily generate from same model
    # if model_name == 'vanilla':
    #     response = temp_qa_chain.run(user_input[-1]['content'])
    # else:
    #     response = temp_qa_chain.run(user_input[-1]['content'])

    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)

def d6(count):
    return random.randint(count, count*6)

if page == 'Character Creator':
    st.header('Create a New Character', divider='gray')

    with st.form("new_character_form"):
        player_name = st.text_input("Player Name")
        char_name = st.text_input("Character Name")
        char_race = st.selectbox("Race", CHARACTER_RACES)
        char_class = st.selectbox("Class", CHARACTER_CLASSES)
        background = st.selectbox("Background", CHARACTER_BACKGROUND)
        stat_list = [15, 14, 13, 12, 10, 8]
        random.shuffle(stat_list)
        stats = f"STR {stat_list[0]}, DEX {stat_list[1]}, CON {stat_list[2]}, INT {stat_list[3]}, WIS {stat_list[4]}, CHA {stat_list[5]}"
        submitted = st.form_submit_button("Add character")

        if submitted:
            if not all([player_name, char_name, char_race, char_class]):
                st.warning("Please fill in all the fields before submitting.")
            else:
                content = f"Player: {player_name}\nName: {char_name}\nRace: {char_race}\nClass: {char_class}\nBackground: {background}\nStats: {stats}\nLevel: 1"
                doc = Document(page_content=content, metadata={"player": player_name, "name": char_name})
                st.session_state['vectorstore'].add_documents([doc])
                st.session_state['players'].append(player_name)
                st.success(f"Character {char_name} of {player_name} added to memory")
    
    st.subheader("Characters")
    if st.session_state['vectorstore'].index.ntotal > 0:
        for i, doc in enumerate(st.session_state['vectorstore'].docstore._dict.values(), start=1):
            st.markdown(f"**{i}. {doc.metadata.get('name', 'Unnamed')}**")
            st.code(doc.page_content.strip())
    else:
        st.info("No character yet. Create one above.")
    
elif page == 'DM Chat':
    st.header('Dungeons and Dragons', divider="gray")

    n = len(st.session_state['history'])
    for i, msg in enumerate(st.session_state['history']):
        if not st.session_state['last_vote_submitted'] and i == n -2:
            break
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    temperature, top_p, top_k = render_sidebar(st.session_state)

    if not st.session_state['last_vote_submitted'] and st.session_state['last_interaction']:
        prompt, res_a, res_b = st.session_state['last_interaction']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model A**")
            with st.chat_message("assistant"):
                st.write(res_a)
            
        with col2:
            st.markdown("**Model B**")
            with st.chat_message("assistant"):
                st.write(res_b)
        show_vote_ui(prompt, res_a, res_b)
        st.chat_input("Say something", disabled=True)
    elif st.session_state['last_vote_submitted']:
        if prompt := st.chat_input("Say something"):
            st.session_state['last_vote_submitted'] = False
            prompt = f"{cur_player}: " + prompt

            with st.chat_message("user"):
                st.write(prompt)
            st.session_state['history'].append({"role": "user", "content": prompt})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Model A**")
                with st.chat_message("assistant"):
                    res_a = st.write_stream(chat_stream(st.session_state['history'], "vanilla", temperature, top_k, top_p))
                
            with col2:
                st.markdown("**Model B**")
                with st.chat_message("assistant"):
                    res_b = st.write_stream(chat_stream(st.session_state['history'], "trained", temperature, top_k, top_p))
            
            st.session_state['last_interaction'] = (prompt, res_a, res_b)
            
            st.session_state['history'].append({"role": "assistant", "content": "".join(res_a)})
            st.session_state['history'].append({"role": "assistant", "content": "".join(res_b)})

            st.rerun()
    else:
        st.warning("Please vote on the last response before continuing")
        st.chat_input("Say something", disabled=True)

# print('------------------------------------------------')
# print(st.session_state)
# print('------------------------------------------------')