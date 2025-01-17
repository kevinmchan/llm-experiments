import os

import torch.distributed as dist
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# Define the model and adapter paths
original_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

#TODO: update it to your chosen epoch
dir_path = os.path.dirname(os.path.realpath(__file__))
trained_model_path = os.path.join(dir_path, "../output/torchtune/qwen2_5_0_5B/lora_single_device/epoch_19")

# llm = LLM(model=original_model_name)

llm = LLM(model=original_model_name, enable_lora=True, max_lora_rank=32)

sampling_params = SamplingParams(max_tokens=16, temperature=0.5)

conversation = [
    {"role": "system", "content": "You are an unhelpful, incorrect math assistant"},
    {"role": "user", "content": "What is 2 + 2?"},
]

lora = LoRARequest("unhelpful", 1, trained_model_path)

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)

outputs = llm.chat(
    conversation,
    sampling_params=sampling_params,
    lora_request=lora,
)

print_outputs(outputs)


# clean up
dist.destroy_process_group()