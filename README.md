# Description

This is telegram bot for cloudnet.

# Configuration

## Environment variables

- TGBOT_API_TOKEN
- TGBOT_MODEL_PATH

# Development

## Setup

```bash
python3 -m venv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
env $(cat .env | xargs) ./service.py
```
