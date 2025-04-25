import torch

from lightning import Fabric
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModelForCausalLM

student_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

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

student_model = AutoModelForCausalLM.from_pretrained(student_model_name, device_map="auto", quantization_config=quant_config,)
tokenizer = AutoTokenizer.from_pretrained(student_model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

for param in student_model.parameters():
  param.requires_grad = False

student_model.gradient_checkpointing_enable()
student_model.enable_input_require_grads()

lora_model = get_peft_model(student_model, lora_config)

fabric = Fabric(accelerator="cuda", devices=1, precision="bf16")
device = fabric.device
fabric.launch()
torch.set_float32_matmul_precision("medium")

model = PeftModelForCausalLM.from_pretrained(lora_model, "./best")

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
       assistant = student_model

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
