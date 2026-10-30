#!/bin/bash
# HLPD inference checkpoint (LoRA adapter on GPT-Neo-2.7B, a few MB) -> ./models/HLPD-inference
# Requires the base model from scripts/download_model.sh.
# The adapter will be published on Hugging Face; you can also train and export it yourself with
#   bash scripts/train_hlpo.sh && bash scripts/export_inference_ckpt.sh <ckpt dir>
HF_REPO=${HF_REPO:-dfq2021/HLPD-inference}
huggingface-cli download "$HF_REPO" --local-dir models/HLPD-inference
