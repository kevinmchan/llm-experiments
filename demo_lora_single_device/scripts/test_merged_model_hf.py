import os

from transformers import AutoModelForCausalLM, AutoTokenizer


#TODO: update it to your chosen epoch
dir_path = os.path.dirname(os.path.realpath(__file__))
merged_model_path = os.path.join(dir_path, "../output/torchtune/qwen2_5_0_5B/lora_single_device/epoch_19")

merged_model = AutoModelForCausalLM.from_pretrained(merged_model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

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

print("Base model output:\n", generate_text(merged_model, tokenizer, prompt))