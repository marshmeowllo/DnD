import streamlit as st
import time
import os
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

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
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embedding = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-modernbert-base",model_kwargs={'trust_remote_code': True})
    index = faiss.IndexFlatL2(len(embedding.embed_query("test")))
    st.session_state['vectorstore'] = FAISS(embedding_function=embedding, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
if "player_data" not in st.session_state:
    st.session_state['player_data'] = {}


temp_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_keys=os.getenv('GOOGLE_API_KEY'),  temperature=0.7)

st.sidebar.title("D&D Dungeon Master")
page = st.sidebar.radio("Go to", ["Character Creator", "DM Chat"])

def chat_stream(user_input, model_name):
    temp_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are a Dungeon Master AI. Use the following character information to help answer the player's question.

    Character Info:
    {context}

    Question: {question}
    Answer:"""
    )
    retriever = st.session_state['vectorstore'].as_retriever()
    temp_qa_chain = RetrievalQA.from_chain_type(
        llm=temp_llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": temp_prompt}
    )

    # response = model_loader.generate_response_with_role(temperature, top_p, top_k, model_name=model_name, user_input=user_input)
    if model_name == 'vanilla':
        response = temp_qa_chain.run(user_input[-1]['content'])
    else:
        response = temp_qa_chain.run(user_input[-1]['content'])

    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)

if page == 'Character Creator':
    st.header('Create a New Character', divider='gray')

    with st.form("new_character_form"):
        player_name = st.text_input("Player Name")
        char_name = st.text_input("Character Name")
        char_race = st.text_input("Race")
        char_class = st.text_input("Class")
        submitted = st.form_submit_button("Add character")

        if submitted:
            content = f"Player: {player_name}\nName: {char_name}\nRace: {char_race}\nClass: {char_class}\n"
            doc = Document(page_content=content, metadata={"player": player_name, "name": char_name})
            st.session_state['vectorstore'].add_documents([doc])
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