#!/bin/bash
# HLPD on the single-task benchmark (Tables 1, 5, 6, 7, 9, 10, 11) with the training checkpoint
# (scoring + reference model, ~40 GB).  For the lightweight route use scripts/eval_all.sh.
# Usage: bash scripts/eval_hlpd_single_task.sh [rewrite|polish|expand|generation]
TASK=${1:-rewrite}
CKPT=${CKPT:-ckpt/hlpd_multitask_hlpo_lr_0.0001_beta_0.05_a_1}

for BASEMODEL in gpt-3.5-turbo gpt-4o Qwen2-7B Llama-3-8B Mistral-7B Deepseek-7b; do
    for FILE in data/single_task/$TASK/$BASEMODEL/*.raw_data.json; do
        [ -e "$FILE" ] || continue
        python -m hlpd.train --eval_only \
            --from_pretrained $CKPT --cache_dir ./models \
            --eval_dataset "$FILE" \
            --output_dir results/single_task/$TASK/$BASEMODEL
    done
done
python tools/show_results.py --result_path results/single_task/$TASK --table
