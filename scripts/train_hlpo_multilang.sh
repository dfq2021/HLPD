#!/bin/bash
# Multilingual experiments (Fig. 8): train on 500 pairs / evaluate on 150 pairs per language.
for LANGUAGE in Chinese Portuguese Spanish; do
    python -m hlpd.train \
        --task_name hlpd_${LANGUAGE}_500 \
        --train_dataset data/multilingual/${LANGUAGE}_train_500_rewrite_gpt-3.5-turbo.raw_data.json \
        --eval_dataset  data/multilingual/${LANGUAGE}_test_150_rewrite_gpt-3.5-turbo.raw_data.json \
        --datanum 500 --epochs 2 --lr 1e-4 --beta 0.05 \
        --output_dir results/multilingual/${LANGUAGE}
done
