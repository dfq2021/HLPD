#!/bin/bash
# Convert a training checkpoint into the lightweight inference checkpoint used by
# evaluate / detect / the web app.  Works for adapters written by hlpd.train and for legacy
# full-state checkpoints (ours-finetuned.pth).
CKPT=${1:-ckpt/hlpd_multitask_hlpo_lr_0.0001_beta_0.05_a_1}
python -m hlpd.export_inference_ckpt --checkpoint "$CKPT" --output models/HLPD-inference --portable_base
