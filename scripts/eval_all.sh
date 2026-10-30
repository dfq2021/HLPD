#!/bin/bash
# Fast evaluation of the released inference checkpoint on every benchmark (~11 GB GPU memory).
# The model is loaded once per call; results go to ./evaluation/<benchmark>/<tag>.hlpd.json
CKPT=${CKPT:-models/HLPD-inference}

python -m hlpd.evaluate --checkpoint $CKPT --output_dir evaluation/adversarial_multitask \
    --eval_dataset "data/adversarial_multitask/*/*.raw_data.json"

python -m hlpd.evaluate --checkpoint $CKPT --output_dir evaluation/advanced_generation \
    --eval_dataset "data/advanced_generation/*/*.raw_data.json"

for TASK in rewrite polish expand generation; do
    python -m hlpd.evaluate --checkpoint $CKPT --output_dir evaluation/single_task/$TASK \
        --eval_dataset "data/single_task/$TASK/*/*.raw_data.json"
done

python -m hlpd.evaluate --checkpoint $CKPT --output_dir evaluation/iterative_revision \
    --eval_dataset "data/iterative_revision/*/*.raw_data.json"

python -m hlpd.evaluate --checkpoint $CKPT --output_dir evaluation/truncation \
    --eval_dataset "data/truncation/polish/*/*.raw_data.json" "data/truncation/adversarial_multitask/*/*.raw_data.json"

python tools/show_results.py --result_path evaluation --table
