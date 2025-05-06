import torch
import uuid
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace
from langchain.tools import tool

from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated, Any, Dict, Optional, TypedDict, Union, List
from lightning import Fabric

from IPython.display import display, Image
from langchain_core.tools import tool
import streamlit as st

from src.utils.initialization import load_llm

# torch.set_float32_matmul_precision("medium")
# fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
# device = fabric.device
# fabric.launch()

embed_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

vector_store = FAISS.load_local(
    "./examples/faiss_spell_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

@tool
def spell_retrieve(query: str) -> str:
    """Retrieve information about dungeons and dragons spell.
    
    Args:
        query (str): The spell name to search for.

    Returns:
        str: The spell information.
    """
    retrieved_docs = vector_store.similarity_search(query, k=3)

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

tools = [
    spell_retrieve,
    user
]

class State(TypedDict):
    name: str
    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    temperature: float
    top_p: float
    top_k: int

class ToolCallRequest(TypedDict):
    """A typed dict that shows the inputs into the invoke_tool function."""

    name: str
    arguments: Dict[str, Any]

class ToolCalling():
    def __init__(self, model_name: str, tools: list[BaseTool]):
        # self.model_name = model_name
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")
        # self.pipe = pipeline(
        #     task="text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     return_full_text=False,
        #     max_new_tokens=512,
        #     top_k=10,
        #     device_map="auto"
        # )
        # self._llm = HuggingFacePipeline(pipeline=self.pipe)
        # self.chat = ChatHuggingFace(llm=self._llm, tokenizer=self.tokenizer)
        self.chat = load_llm()

        self.rendered_tools = [convert_to_openai_tool(f) for f in tools]

    def invoke_tool(
            self,
            tool_call_request: Union[ToolCallRequest, List[ToolCallRequest]], 
            config: Optional[RunnableConfig] = None
    ):
        """A function that we can use the perform a tool invocation.

        Args:
            tool_call_request: a dict that contains the keys name and arguments.
                The name must match the name of a tool that exists.
                The arguments are the arguments to that tool.
            config: This is configuration information that LangChain uses that contains
                things like callbacks, metadata, etc.See LCEL documentation about RunnableConfig.

        Returns:
            output from the requested tool
        """

        print("Tool call request:", tool_call_request)
        
        # Sometimes the model outputs a list of tool call requests, 
        # so I loop each tool call and append to list
        
        if isinstance(tool_call_request, list):
            output = list()

            for tool_call in tool_call_request:
                tool_name_to_tool = {tool.name: tool for tool in tools}
                name = tool_call["name"]
                requested_tool = tool_name_to_tool[name]
                output.append(requested_tool.invoke(tool_call["arguments"], config=config))
            return output
        
        tool_name_to_tool = {tool.name: tool for tool in tools}
        name = tool_call_request["name"]
        requested_tool = tool_name_to_tool[name]
        return requested_tool.invoke(tool_call_request["arguments"], config=config)
    
    def invoke(self, state: State) -> List[str]:
        system_prompt = SystemMessage(f"""\
        You are an assistant that has access to the following set of tools. 
        Here are the names and descriptions for each tool:

        {self.rendered_tools}

        Given the user input, return the name and input of the tool to use. 
        Return your response as a JSON blob with 'name' and 'arguments' keys.

        The `arguments` should be a dictionary, with keys corresponding 
        to the argument names and the values corresponding to the requested values.
        """)

        chain = self.chat | JsonOutputParser() | self.invoke_tool
        messages = [system_prompt] + [state['messages'][-1]]

        try:
            response = chain.invoke(messages)
        except Exception as e:
            print("Error invoking tool:", e)
            response = ["No tool needed"]

        if not response:
            response = ["No tool needed"]
        
        return {"context": AIMessage(response)}

tool_calling_model_name = "Salesforce/Llama-xLAM-2-8b-fc-r"

tool_calling = ToolCalling(
    model_name=tool_calling_model_name,
    tools=tools
)

class LlamaChat():
    def __init__(self, model_name: str):
        # self.model_name = model_name
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, load_in_4bit=True)
        # self.pipe = pipeline(
        #     task="text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     return_full_text=False,
        #     max_new_tokens=2048,
        #     top_k=10,
        #     device_map="auto"
        # )
        # self._llm = HuggingFacePipeline(pipeline=self.pipe)
        self.chat = load_llm()

    def generate(self, state: State) -> Dict[str, Any]:
        system_message_content = (
            "<|start_header_id|>system<|end_header_id|>\n"
            "In a text-based adventure (Dungeons and Dragons), your job is to narrate the adventure "
            "and respond to the player's actions.\n"
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer, say i dont know."
            "If the player breaks the game rules, "
            "notify the player.\n"
            "This is the retrieved context:\n\n"
            f"{state['context']}\n\n"
            "When you answer the player, you must respond in proper markdown format: heading, table, bold, italic, paragraph, blockquotes.\n"
        )

        # print('State:', state)
        # print('Context:', state["context"])
        # print('Generating response ...')

        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]

        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # chat = ChatHuggingFace(llm=self._llm, tokenizer=self.tokenizer, pipeline_kwargs={ "temperature": state['temperature'], "top_k": state['top_k'], "top_p": state['top_p']})
        # chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_keys=os.getenv('GOOGLE_API_KEY'),  temperature=0.7)

        # response = chat.invoke(prompt)
        response = self.chat.invoke(prompt)
        
        return {"messages": [response]}
    
llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

llama = LlamaChat(model_name=llama_model_name)

def tool(state: State):
    return tool_calling.invoke(state)

def chatbot(state: State) -> str:
    return llama.generate(state=state)

memory = MemorySaver()

graph = StateGraph(State)

graph.add_edge(START, "tool call")
graph.add_node("tool call", tool)

graph.add_edge("tool call", "chatbot")

graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)

graph = graph.compile(checkpointer=memory)

def generate_response(player_name, prompt, temperature, top_p, top_k, model_name):
    text = f"<|start_header_id|>{player_name}<|end_header_id|>\n{prompt}<|eot_id|>"

    input_state = {
        "name": player_name,
        "messages": [
            HumanMessage(content=text)
        ],
        "context": "",
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }

    if model_name == 'vanilla':
        response = graph.invoke(input_state, config={"configurable": {"thread_id": uuid.uuid4()}})
    else:
        response = graph.invoke(input_state, config={"configurable": {"thread_id": uuid.uuid4()}})

    # print("Response:", response)

    return response['messages'][-1].content
