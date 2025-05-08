import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.tools import tool

from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langgraph.graph.message import add_messages

from typing import Annotated, Any, Dict, Optional, TypedDict, Union, List
from lightning import Fabric

from src.tools.tools import spell_retrieve, user
from src.utils.initialization import load_llm

# torch.set_float32_matmul_precision("medium")
# fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
# device = fabric.device
# fabric.launch()

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

tools=[spell_retrieve, user]

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

        # response = chat.invoke(prompt)
        response = self.chat.invoke(prompt)
        
        return {"messages": [response]}

def tool(state: State):
    return st.session_state['tool_calling'].invoke(state)

def chatbot(state: State) -> str:
    return st.session_state['llama'].generate(state=state)

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

    response = st.session_state['graph'].invoke(input_state, config={"configurable": {"thread_id": model_name}})

    return response['messages'][-1].content
