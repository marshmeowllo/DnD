import random
import streamlit as st
from langchain_core.documents import Document
from config import CHARACTER_BACKGROUND, CHARACTER_CLASSES, CHARACTER_RACES

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