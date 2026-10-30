#!/bin/bash
# Base scoring / reference model (GPT-Neo-2.7B) -> ./models
# export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
huggingface-cli download EleutherAI/gpt-neo-2.7B --exclude "*.bin" "*.ot" "*.msgpack" --local-dir models/gpt-neo-2.7B
