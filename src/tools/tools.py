from langchain_core.tools import tool
import streamlit as st

@tool
def spell_retrieve(query: str) -> str:
    """Retrieve information about dungeons and dragons spell.
    
    Args:
        query (str): The spell name to search for.

    Returns:
        str: The spell information.
    """
    retrieved_docs = st.session_state['spellstore'].similarity_search(query, k=3)

    contents = "\n\n".join(
        (f"{doc.page_content}")
        for doc in retrieved_docs
    )
    
    return contents

@tool
def user(name: str) -> str:
    """
    User infomation retreiver

    Args:
        name (str): The name of user.

    Returns:
        str: The user information.
    """
    retrieved_docs = st.session_state['vectorstore'].similarity_search(name, k=3)

    contents = "\n\n".join(
        (f"{doc.page_content}")
        for doc in retrieved_docs
    )
    
    return contents