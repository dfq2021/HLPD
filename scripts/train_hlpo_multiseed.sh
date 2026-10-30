#!/bin/bash
# Five independent runs (seeds 42 199 410 2231 2533) evaluated on the four Gemini-2.5-Pro
# adversarial sets -> mean AUROC with 95% CI (Appendix A, Fig. 4/5).
python -m hlpd.train_multiseed \
    --task_name hlpd_multitask \
    --seeds 42 199 410 2231 2533 \
    --eval_datasets \
        data/adversarial_multitask/gemini-2.5-pro/xsum_adversarial_gemini-2.5-pro.raw_data.json \
        data/adversarial_multitask/gemini-2.5-pro/writing_adversarial_gemini-2.5-pro.raw_data.json \
        data/adversarial_multitask/gemini-2.5-pro/squad_adversarial_gemini-2.5-pro.raw_data.json \
        data/adversarial_multitask/gemini-2.5-pro/wiki_adversarial_gemini-2.5-pro.raw_data.json \
    --output_dir results/multi_seed/gemini-2.5-pro

python tools/plot_roc_ci.py --runs results/multi_seed/gemini-2.5-pro --output figures/roc_ci_gemini-2.5-pro.pdf
