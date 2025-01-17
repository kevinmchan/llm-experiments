# Demo Lora Single Device Fine Tuning

Can we demonstrate that fine tuning using LoRA is working successfully?

Let's fine tune a QWEN 2.5 Instruct model to unhelpfully give the incorrect answer to math questions.


## Reference materials

- [LoRA recipe](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html#lora-finetune-label)
- [Fine tune end-to-end usage](https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html)
- [vLLM Lora usage](https://docs.vllm.ai/en/latest/features/lora.html)


## Setup environment

Setup environment using uv.lock file:
```bash
uv sync
```

Activate created environment:
```bash
source .venv/bin/activate
```


## Fine tuning proceedure

- [x] Generate boilerplate config
```bash
mkdir config
tune cp qwen2_5/0.5B_lora_single_device config/qwen_config.yaml
```

- [x] Generate boilerplate recipe
```bash
mkdir recipes
touch recipes/__init__.py
tune cp lora_finetune_single_device recipes/lora_single_device.py
```

- [x] Download base model
```bash
mkdir model
tune download Qwen/Qwen2.5-0.5B-Instruct --output-dir ./model/Qwen2.5-0.5B-Instruct
```

- [x] Create output directory
```bash
mkdir output
```

- [x] Create sample data in `/data/test.json`

- [x] Update config
    - [x] Update location of base model
    - [x] Update output directory
    - [x] Update data component to use chat dataset builder
    - [x] Set to save only adapter weights

- [x] Update config and recipe to allow checkpointing on nth epoch

- [x] Run fine tuning recipe
```bash
PYTHONPATH=${pwd}:PYTHONPATH tune run recipes/lora_single_device.py --config config/qwen_config.yaml
```


## Running inference with adapters

Using huggingface's transformers:
```bash
uv run --with peft --no-project scripts/test_adapter_hf.py
```

Using vllm:
```bash
uv run --with vllm --no-project scripts/test_adapter_vllm.py
```

Note that in both cases above, we use `uv run` to avoid installing peft and vllm which are not required for our training recipe.