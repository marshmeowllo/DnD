import streamlit as st

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from config import MODEL_MAP
from src.models.model import LlamaChat, ToolCalling, State, tool, chatbot
from src.tools.tools import spell_retrieve, user

def build_graph(memory):
    graph = StateGraph(State)
    graph.add_edge(START, "tool call")
    graph.add_node("tool call", tool)
    graph.add_edge("tool call", "chatbot")
    graph.add_node("chatbot", chatbot)
    graph.add_edge("chatbot", END)
    return graph.compile(checkpointer=memory)

def init_models(model_choice: str):
    model_name = MODEL_MAP[model_choice]
    st.session_state['history'] = []
    st.session_state['tool_calling'] = ToolCalling(model_name=model_name, tools=[spell_retrieve, user])
    st.session_state['llama'] = LlamaChat(model_name=model_name)
    st.session_state['memory'] = MemorySaver()
    st.session_state['graph'] = build_graph(st.session_state['memory'])