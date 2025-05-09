import streamlit as st
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.models.model import State, tool, chatbot

def build_graph(memory):
    graph = StateGraph(State)
    graph.add_edge(START, "tool call")
    graph.add_node("tool call", tool)
    graph.add_edge("tool call", "chatbot")
    graph.add_node("chatbot", chatbot)
    graph.add_edge("chatbot", END)
    return graph.compile(checkpointer=memory)

def change_model():
    st.session_state['history'] = []
    st.session_state['memory'] = MemorySaver()
    st.session_state['graph'] = build_graph(st.session_state['memory'])