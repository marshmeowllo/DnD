import streamlit as st

from config import CHARACTER_BACKGROUND, CHARACTER_CLASSES, CHARACTER_RACES
from src.utils.character import create_character_doc, roll_ability_scores

st.header('Create a New Character', divider='gray')

if 'vectorstore' not in st.session_state:
    st.error("Session state not initialized. Please return to the main page.")
    st.stop()

if 'players' not in st.session_state:
    st.session_state['players'] = []

with st.form("new_character_form"):
    char_name = st.text_input("Character Name")
    char_race = st.selectbox("Race", CHARACTER_RACES)
    char_class = st.selectbox("Class", CHARACTER_CLASSES)
    background = st.selectbox("Background", CHARACTER_BACKGROUND)
    stat_list = roll_ability_scores()
    stats = f"STR {stat_list[0]}, DEX {stat_list[1]}, CON {stat_list[2]}, INT {stat_list[3]}, WIS {stat_list[4]}, CHA {stat_list[5]}"
    submitted = st.form_submit_button("Add character")

    if submitted:
        if not all([char_name, char_race, char_class]):
            st.warning("Please fill in all the fields before submitting.")
        else:
            content = f"Name: {char_name}\nRace: {char_race}\nClass: {char_class}\nBackground: {background}\nStats: {stats}\nLevel: 1"
            doc = create_character_doc(char_name,char_race,char_class,background,stats)
            st.session_state['vectorstore'].add_documents([doc])
            if char_name not in st.session_state['players']:
                st.session_state['players'].append(char_name)
            st.success(f"Character {char_name} added to memory")

st.subheader("Characters")
if st.session_state['vectorstore'].index.ntotal > 0:
    for i, doc in enumerate(st.session_state['vectorstore'].docstore._dict.values(), start=1):
        st.markdown(f"**{i}. {doc.metadata.get('name', 'Unnamed')}**")
        st.code(doc.page_content.strip())
else:
    st.info("No character yet. Create one above.")