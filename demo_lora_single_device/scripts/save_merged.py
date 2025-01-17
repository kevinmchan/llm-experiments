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

# merge models
merged_model = peft_model.merge_and_unload()

# save model
merged_model_path = os.path.join(dir_path, "../merged_model")
merged_model.save_pretrained(merged_model_path)


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(original_model_name)

# save tokenizer
tokenizer.save_pretrained(merged_model_path)