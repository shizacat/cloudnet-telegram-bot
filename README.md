# Description

This is telegram bot for cloudnet.

# Development

## Setup

```bash
python3 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
env $(cat .env | xargs) ./service.py --model-path ../../cloudnet-web/contribute/model.onnx
```
