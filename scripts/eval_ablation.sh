#!/bin/bash
# Ablations of Table 4 on the adversarial multi-task benchmark (DeepSeek-R1 / Gemini-2.5-Pro).
#   (a) objective: sigmoid vs linear (‡), fixed vs Dynamic-beta (⋆)
#   (b) alignment strategy: SFT / ORPO / RLHF (human-preferred) vs HLPO
EVALS="data/adversarial_multitask/deepseek-r1/*.raw_data.json data/adversarial_multitask/gemini-2.5-pro/*.raw_data.json"
MONITOR=data/adversarial_multitask/gemini-2.5-pro/xsum_adversarial_gemini-2.5-pro.raw_data.json

# ---- (a) objective ablations: train, then evaluate on the eight adversarial sets ------------
run_variant () {  # name, extra train args
    NAME=$1; shift
    python -m hlpd.train --task_name $NAME "$@" --eval_dataset $MONITOR \
        --ckpt_dir ckpt/ablation_$NAME --output_dir results/ablation/$NAME
    for FILE in $EVALS; do
        python -m hlpd.train --eval_only --from_pretrained ckpt/ablation_$NAME \
            --eval_dataset "$FILE" --output_dir results/ablation/$NAME/$(basename $(dirname $FILE))
    done
}
run_variant hlpo_sigmoid_fixed --loss sigmoid --no_dynamic_beta   # DPO/ImBD-style objective (human-preferred)
run_variant hlpo_sigmoid_dyn   --loss sigmoid                     # HLPO⋆
run_variant hlpo_linear_fixed  --loss linear  --no_dynamic_beta   # HLPO‡
run_variant hlpo_linear_dyn    --loss linear                      # HLPO‡⋆ (full method)

# ---- (b) alignment-strategy ablations -------------------------------------------------------
python ablation_exp/train_gpt_neo_sft.py  --prefer human --datanum 3000 --save_name hlpd_sft_human_3000
python ablation_exp/train_gpt_neo_orpo.py --prefer human --save_name hlpd_orpo_human
python ablation_exp/train_gpt_neo_rlhf.py --prefer human --save_name hlpd_rlhf_human
for NAME in hlpd_sft_human_3000 hlpd_orpo_human hlpd_rlhf_human; do
    for FILE in $EVALS; do
        # fully fine-tuned checkpoints are loaded as the base model (no extra LoRA)
        python -m hlpd.train --eval_only --no_lora --base_model $NAME --cache_dir ./ckpt \
            --eval_dataset "$FILE" --output_dir results/ablation/$NAME/$(basename $(dirname $FILE))
    done
done
python tools/show_results.py --result_path results/ablation --table
