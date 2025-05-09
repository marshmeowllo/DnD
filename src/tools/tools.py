import streamlit as st
from typing import List

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

@tool
def retrieve(query: str) -> str:
    """Retrieve information about dungeons and dragons: classes, item, monsters, races, and spells.
    
    Args:
        query (str): name or thing to search for.

    Returns:
        str: The retrieve information.
    """
    retrieved_docs = st.session_state['dnd'].similarity_search(query, k=3)

    contents = "\n\n".join(
        (f"{doc.page_content}")
        for doc in retrieved_docs
    )
    
    return contents

@tool
def user(name: str) -> str:
    """
    User information retreiver

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

def get_openai_tools() -> List[dict]:
    functions = [retrieve, user]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools