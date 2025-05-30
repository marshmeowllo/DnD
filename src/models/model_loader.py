import torch
import os

from lightning import Fabric
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModelForCausalLM

MODEL_DIR = os.path.join(os.path.dirname(__file__), "best")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

torch.set_float32_matmul_precision("medium")
fabric = Fabric(accelerator="cuda", devices=1, precision="bf16")
device = fabric.device
fabric.launch()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM"
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
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    is_trainable=False
    )

model = model.eval()
model.config.use_cache = True

def generate_response_with_role(temperature, top_p, top_k, model_name="Trained", user_input='', max_length=4096):
    messages = [
      {"role": "system", "content": "in a text-based adventure (Dungeons and Dragons).\nYour job is to narrate the adventure and respond to the player's actions.\nWhen you anwser to player you must answer in proper markdown format. (heading, table, bold, italic, paragraph, blockquotes)"},
    ]

    for converstion in user_input:
      messages.append(converstion)

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    if model_name == "Trained":
       assistant = model
    else:
       assistant = base_model

    with torch.no_grad():
        outputs = assistant.generate(
            input_ids,
            max_length=max_length,
            eos_token_id=terminators,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)

    return response