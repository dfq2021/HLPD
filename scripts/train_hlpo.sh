#!/bin/bash
# HLPO training (paper setting): GPT-Neo-2.7B + LoRA, 2 epochs, lr 1e-4, beta_0 = 0.05,
# linear objective + Dynamic-beta, all 3,354 multi-task revision pairs.
# ~40 GB GPU memory (scoring + reference model), ~12 min / epoch on an L20.
python -m hlpd.train \
    --task_name hlpd_multitask \
    --train_dataset data/train/train_multitask_revision.raw_data.json \
    --eval_dataset data/adversarial_multitask/gpt-4o/xsum_adversarial_gpt-4o.raw_data.json \
    --datanum -1 --epochs 2 --lr 1e-4 --beta 0.05 \
    --loss linear \
    --output_dir results/train_monitor
# -> ckpt/hlpd_multitask_hlpo_lr_0.0001_beta_0.05_a_1   (LoRA adapter)
