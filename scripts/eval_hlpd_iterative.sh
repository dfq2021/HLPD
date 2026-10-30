#!/bin/bash
# Robustness to 1..5 revision iterations (Table 8).
CKPT=${CKPT:-ckpt/hlpd_multitask_hlpo_lr_0.0001_beta_0.05_a_1}
for BASEMODEL in gpt-4o gemini-2.0-flash-exp gpt-4o-mini; do
    for FILE in data/iterative_revision/$BASEMODEL/*.raw_data.json; do
        python -m hlpd.train --eval_only \
            --from_pretrained $CKPT --cache_dir ./models \
            --eval_dataset "$FILE" \
            --output_dir results/iterative_revision/$BASEMODEL
    done
done
python tools/show_results.py --result_path results/iterative_revision
