#!/bin/bash
# Models used by the baselines (Fast-DetectGPT, DetectGPT, NPR, RoBERTa detectors).
# export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
HF_TOKEN=${HF_TOKEN:-}   # needed for gated generation models below

huggingface-cli download EleutherAI/gpt-j-6b --exclude "*.ot" "*.msgpack" "*.h5" --local-dir models/gpt-j-6B
huggingface-cli download google-t5/t5-3b --exclude "*.bin" "*.h5" --local-dir models/t5-3b
huggingface-cli download openai-community/roberta-large-openai-detector --exclude "*.bin" "*.msgpack" --local-dir models/roberta-large-openai-detector
huggingface-cli download openai-community/roberta-base-openai-detector --exclude "*.bin" "*.h5" "*.msgpack" --local-dir models/roberta-base-openai-detector

# Open-source generators used to (re)build the single-task benchmark (scripts/build_data_opensource.sh)
# huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir models/Qwen2-7B-Instruct
# huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --exclude consolidated.safetensors --local-dir models/Mistral-7B-Instruct-v0.3 --token $HF_TOKEN
# huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "*.pth" --local-dir models/Meta-Llama-3-8B-Instruct --token $HF_TOKEN
# huggingface-cli download deepseek-ai/deepseek-llm-7b-chat --local-dir models/deepseek-llm-7b-chat
