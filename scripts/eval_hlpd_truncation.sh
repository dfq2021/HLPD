#!/bin/bash
# Passage-length robustness (Fig. 9 polish, Fig. 10 adversarial multi-task): 30..180 words.
CKPT=${CKPT:-ckpt/hlpd_multitask_hlpo_lr_0.0001_beta_0.05_a_1}
for FILE in data/truncation/polish/*/*.raw_data.json data/truncation/adversarial_multitask/*/*.raw_data.json; do
    BENCH=$(basename $(dirname $(dirname $FILE)))
    BASEMODEL=$(basename $(dirname $FILE))
    python -m hlpd.train --eval_only \
        --from_pretrained $CKPT --cache_dir ./models \
        --eval_dataset "$FILE" \
        --output_dir results/truncation/$BENCH/$BASEMODEL
done
python tools/show_results.py --result_path results/truncation
