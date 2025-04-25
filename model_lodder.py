import torch
import os
import uuid

from typing import Any, Dict, Iterator, List, Optional, TypedDict

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain.schema import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from lightning import Fabric
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM, PeftModel

torch.set_float32_matmul_precision("medium")
fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
device = fabric.device
fabric.launch()

embed_model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
vector_store = FAISS.load_local(
    "./utils/faiss_spell_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever()

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM"
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    )

lora_model = get_peft_model(base_model, lora_config)

model = PeftModelForCausalLM.from_pretrained(
    lora_model, 
    "./best",
    torch_dtype=torch.bfloat16,
    is_trainable=False
    )

model = model.eval()
model.config.use_cache = True

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=4096,
    top_k=50,
    device_map="auto"
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

llm = HuggingFacePipeline(
    pipeline=pipe, 
    model_kwargs = {'temperature': 0.9, "torch_dtype": torch.bfloat16}
    )

chat_llama3 = ChatHuggingFace(llm=llm)
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = chat_llama3.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "<|start_header_id|>system<|end_header_id|>\n"
        "In a text-based adventure (Dungeons and Dragons), your job is to narrate the adventure "
        "and respond to the player's actions.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, answer based on your knowledge. If the player breaks the game rules, "
        "notify the player.\n"
        f"{MessagesPlaceholder(variable_name='history')}"
        "This is the retrieved context:\n\n"
        f"{docs_content}\n\n"
        "When you answer the player, you must respond in proper markdown format: heading, table, bold, italic, paragraph, blockquotes.\n"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


memory = MemorySaver()
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer=memory)