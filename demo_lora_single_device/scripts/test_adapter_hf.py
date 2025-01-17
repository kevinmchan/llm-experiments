import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define the model and adapter paths
original_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

#TODO: update it to your chosen epoch
dir_path = os.path.dirname(os.path.realpath(__file__))
trained_model_path = os.path.join(dir_path, "../output/torchtune/qwen2_5_0_5B/lora_single_device/epoch_19")

model = AutoModelForCausalLM.from_pretrained(original_model_name)

# huggingface will look for adapter_model.safetensors and adapter_config.json
peft_model = PeftModel.from_pretrained(model, trained_model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(original_model_name)

# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0])

conversation = [
    {"role": "system", "content": "You are an unhelpful, incorrect math assistant"},
    {"role": "user", "content": "What is 2 + 6?"},
]
prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

print("Base model output:\n", generate_text(peft_model, tokenizer, prompt))