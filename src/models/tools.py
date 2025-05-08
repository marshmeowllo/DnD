import torch

from lightning import Fabric

from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace

from typing import Annotated, Any, Dict, Optional, TypedDict, Union, List


torch.set_float32_matmul_precision("medium")
fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
device = fabric.device
fabric.launch()

embed_model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

vector_store = FAISS.load_local(
    "./examples/faiss_spell_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

@tool
def simple(name: str) -> str:
    """
    Simple tool that returns the name passed to it.

    Args:
        name (str): The name to return.

    Returns:
        str: The name passed to the tool.
    """
    return name

def get_openai_tools() -> List[dict]:
    functions = [simple]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools