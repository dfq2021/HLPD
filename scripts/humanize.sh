#!/bin/bash
# Humanization adaptive attack (Sec. 3.5): 100 GPT-3.5 polished XSum passages, 100 T5-3B
# perturbations per iteration, 4 iterations, top-1 selection by the HLPD scorer.
python tools/humanize.py \
    --input data/single_task/polish/gpt-3.5-turbo/xsum_polish_gpt-3.5-turbo.raw_data.json \
    --checkpoint models/HLPD-inference --mask_filling_model_name t5-3b \
    --n 100 --n_perturbations 100 --iterations 4 \
    --output results/humanize/xsum_polish_gpt-3.5-turbo.humanized.json

# Evaluate any detector on the humanized versions, e.g. HLPD itself:
python -m hlpd.evaluate --checkpoint models/HLPD-inference \
    --eval_dataset "results/humanize/*.iter*.raw_data.json" --output_dir evaluation/humanize
