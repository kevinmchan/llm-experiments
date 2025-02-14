# Demo LoRA Multi-Device Fine Tuning

Can we demonstrate that fine tuning using LoRA is working successfully, in a multi-gpu setup?

Let's fine tune a QWEN 2.5 32B Instruct model to unhelpfully give the incorrect answer to math questions.

This demo builds upon [demo_lora_single_device](../demo_lora_single_device/) but instead uses the `lora_finetune_distributed` recipe and `qwen2_5/32B_lora` config.


## Reference materials

- [LoRA recipe](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html#lora-finetune-label)
- [Fine tune end-to-end usage](https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html)
- [vLLM Lora usage](https://docs.vllm.ai/en/latest/features/lora.html)


## Setup environment

If setting up the project environment for the first time:
```bash
uv init --python=3.12
uv add torch torchao torchvision torchtune
```

Alternatively, setup environment using uv.lock file included in the repo:
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
tune cp qwen2_5/32B_lora config/qwen_config.yaml
```

- [x] Generate boilerplate recipe
```bash
mkdir recipes
touch recipes/__init__.py
tune cp lora_finetune_distributed recipes/lora_distributed.py
```

- [x] Download base model
```bash
mkdir model
tune download Qwen/Qwen2.5-32B-Instruct --output-dir ./model/Qwen2_5-32B-Instruct
```

- [x] Create output directory
```bash
mkdir output
```

- [x] Create sample data in `/data/test.json`

- [x] Update config
    - [x] Update output directory
    - [x] Update location of base model checkpoint
    - [x] Update location of tokenizer (path and vocab)
    - [x] Set to save only adapter weights
    - [x] Update data component to use chat dataset builder
    - [x] Update training params

- [x] Run fine tuning recipe
```bash
PYTHONPATH=${pwd}:PYTHONPATH tune run --nproc_per_node 8 recipes/lora_distributed.py --config config/qwen_config.yaml
```