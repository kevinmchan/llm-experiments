import os

from vllm import LLM, SamplingParams


#TODO: update it to your chosen epoch
dir_path = os.path.dirname(os.path.realpath(__file__))
merged_model_path = os.path.join(dir_path, "../merged_model")

# llm = LLM(model=original_model_name)

llm = LLM(model=merged_model_path)

sampling_params = SamplingParams(max_tokens=16, temperature=0.5)

conversation = [
    {"role": "system", "content": "You are an unhelpful, incorrect math assistant"},
    {"role": "user", "content": "What is 2 + 2?"},
]

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)

outputs = llm.chat(
    conversation,
    sampling_params=sampling_params,
)

print_outputs(outputs)
