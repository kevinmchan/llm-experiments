# Testing VLLM environment using UV

Can I create a virtual environment using UV that successfully runs the VLLM package?

## Replication Notes

### Install UV
Install uv per [instructions](https://github.com/astral-sh/uv):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart shell to ensure uv is added to PATH.

Confirm running the latest version:
```bash
uv self update
```

### Initialize project

Initialize project in the working directory:
```bash
uv init
```

### Install vllm

Install vllm package:
```bash
uv add vllm
```

### Run vllm script

Run:
```bash
uv run main.py
```

Success!
