import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM, PeftModel

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages

from typing import Annotated, Any, Dict, Optional, TypedDict, Union, List
from lightning import Fabric

from src.tools.tools import get_openai_tools, retrieve, user

torch.set_float32_matmul_precision("medium")
fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
device = fabric.device
fabric.launch()

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

tools = [retrieve, user]

class ToolCalling():
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu")
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            max_new_tokens=512,
            top_k=10,
            device_map="auto"
        )
        self._llm = HuggingFacePipeline(pipeline=self.pipe)
        self.chat = ChatHuggingFace(llm=self._llm, tokenizer=self.tokenizer)
        # self.chat = load_llm(model_name)

        self.rendered_tools = get_openai_tools()
        
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
        match model_name:
            case 'SFT_Yuaylong':
                self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', torch_dtype=torch.bfloat16, load_in_4bit=True)
                self.model = PeftModel.from_pretrained(self.model, './src/models/weights/SFT_Yuaylong')
                self.model = self.model.merge_and_unload()
            case 'bestRL':
                self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype=torch.bfloat16, load_in_4bit=True)
                self.model = PeftModelForCausalLM.from_pretrained(model, './src/models/weights/bestRL')
            case 'vanilla':
                self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
                self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype=torch.bfloat16, load_in_4bit=True)
            case 'SFT_RL_V1_Son':
                self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
                self.model = AutoModelForCausalLM.from_pretrained('./src/models/weights/SFT_RL_V1_Son', torch_dtype=torch.bfloat16, load_in_4bit=True)                
            case _:
                raise ValueError(f"Model name {model_name} not recognized. Please use 'SFT_Yuaylong', 'bestRL', or 'vanilla'.")

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            max_new_tokens=2048,
            top_k=10,
            device_map="auto"
        )
        self._llm = HuggingFacePipeline(pipeline=self.pipe)

    def generate(self, state: State) -> Dict[str, Any]:
        system_message_content = (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a highly skilled Dungeon Master (DM) for Dungeons & Dragons 5th Edition. "
            "Your job is to read the player’s input and reply with an immersive, clear, and engaging narration that advances the story and game mechanics.\n"
            "\n"
            "Advice for the DM:\n"
            "  - Describe scenes with vivid sensory details and atmosphere.\n"
            "  - Offer meaningful choices if players ask for choices; respect player agency and avoid railroading.\n"
            "  - Keep secret die rolls hidden; reveal only the results and their effects.\n"
            "  - Use rules faithfully but prioritize fun and pacing over strict book-keeping.\n"
            "  - Improvise when players surprise you, but maintain internal consistency.\n"
            "  - Avoid technical jargon in your narration—stay in character and tone.\n"
            "\n"
            "Basic D&D 5e Rules Summary:\n"
            "  - Initiative: roll a d20 + Dexterity modifier to determine turn order.\n"
            "  - Actions: on your turn you can take one Action, one Bonus Action (if available), and move up to your speed.\n"
            "  - Reactions: special actions triggered outside your turn, such as Opportunity Attacks.\n"
            "  - Advantage & Disadvantage: roll two d20s; take the higher roll for advantage or the lower for disadvantage.\n"
            "  - Spellcasting: consumes spell slots; cantrips are cast at will; concentration holds one spell at a time.\n\n"
            "Campaign detail:\n"
            "At the heart of Barovia, a land shrouded in perpetual mist and the oppressive gloom of Castle Ravenloft looming above, the adventurers find themselves trapped within the cursed domain of Strahd.\n"
            "The infamous vampire lord Strahd von Zarovich rules with iron and blood, his tragic past entwined with dark magic and unrequited love.\n"
            "The party’s destiny is guided by a cryptic Tarokka deck reading from Madam Eva, randomizing the locations of three powerful artifacts—the Sunsword, the Tome of Strahd, and the Holy Symbol of Ravenkind—to set the stakes for their final confrontation.\n"
            "Early survival often hinges on navigating the deathly Svalich Woods, where sentient trees and wendigos prowl, and entering the perilous Death House to learn that not all horrors are undead.\n"
            "In the Village of Barovia, the PCs meet Ismark and Ireena Kolyana—siblings haunted by Strahd’s obsession—and may face a night assault in the graveyard by the vampire himself.\n"
            "As they pursue their quest, they reach Vallaki’s forced festivals of joy, unmasking Baron Vallakovich’s dark secrets beneath the town’s brittle gaiety. :\n"
            "The modular nature of the land allows for side trips to Argynvostholt’s haunted halls, Van Richten’s Tower of mad experimentation, and the haunted Ruins of Berez—each offering unique encounters like the Abbot’s twisted deva or Baba Lysaga’s creeping hut. :\n"
            "A crucial detour to the Wizard of Wines winery leads through Yester Hill, where twig blights and corrupted druids guard a dark gem needed to lift the vineyard’s curse. :\n"
            "At their journey’s culmination, the PCs storm Castle Ravenloft’s shadowed halls, confront Strahd amid gothic spires, and challenge his vampire spawn and brides in a climactic battle for Barovia’s freedom. :\n"
            "The module’s gothic ambiance is enriched by haunting music cues, Tarokka card handouts, and richly illustrated maps that deepen immersion in Barovia’s despair. :\n"
            "This is the retrieved context:\n\n"
            f"{state['context']}\n\n"
            "You must follow the these rules strictly\n\n "
            "1. DO NOT simulate the result of player\'s actions.\n\n"
            "2.Do not include <CHARACTER>.\n\n"
            "3.Use the context provided.\n\n"
            "YOU MUST ALSO USE these current game state data :"
            "Current location: Vallaki’s forced festivals of joy"
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

        chat = ChatHuggingFace(llm=self._llm, tokenizer=self.tokenizer, pipeline_kwargs={ "temperature": state['temperature'], "top_k": state['top_k'], "top_p": state['top_p']})

        response = chat.invoke(prompt)
        # response = self.chat.invoke(prompt)
        
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
