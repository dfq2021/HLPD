#!/bin/bash
# HLPD on the adversarial multi-task revision benchmark (Table 3) and on advanced-LLM generations (Table 2).
CKPT=${CKPT:-ckpt/hlpd_multitask_hlpo_lr_0.0001_beta_0.05_a_1}

for BASEMODEL in gpt-4o claude-3.5-sonnet deepseek-r1 gemini-2.5-pro; do
    for DATASET in xsum writing squad wiki; do
        python -m hlpd.train --eval_only \
            --from_pretrained $CKPT --cache_dir ./models \
            --eval_dataset data/adversarial_multitask/$BASEMODEL/${DATASET}_adversarial_$BASEMODEL.raw_data.json \
            --output_dir results/adversarial_multitask/$BASEMODEL
    done
done

for BASEMODEL in claude-3.5-sonnet deepseek-r1 gemini-2.5-pro grok-3; do
    python -m hlpd.train --eval_only \
        --from_pretrained $CKPT --cache_dir ./models \
        --eval_dataset data/advanced_generation/$BASEMODEL/xsum_generation_$BASEMODEL.raw_data.json \
        --output_dir results/advanced_generation/$BASEMODEL
done

python tools/show_results.py --result_path results/adversarial_multitask --table
python tools/show_results.py --result_path results/advanced_generation --table
