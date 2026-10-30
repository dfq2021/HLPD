#!/bin/bash
# Zero-shot baselines (Likelihood, LogRank, Entropy, LRR, NPR, DNA-GPT, DetectGPT, Fast-DetectGPT)
# on one benchmark.  Usage: bash scripts/eval_other_models.sh <benchmark> [source models...]
#   <benchmark>: rewrite | polish | expand | generation | adversarial_multitask | advanced_generation
# Requires scripts/download_other_models.sh (GPT-J for Fast-DetectGPT, T5-3B for DetectGPT / NPR).
BENCH=${1:-rewrite}
shift
case $BENCH in
    adversarial_multitask|advanced_generation) DIR=data/$BENCH ;;
    *) DIR=data/single_task/$BENCH ;;
esac
OUT=other_model_results/$BENCH
MODELS=${@:-$(ls $DIR)}

for BASEMODEL in $MODELS; do
    for FILE in $DIR/$BASEMODEL/*.raw_data.json; do
        [ -e "$FILE" ] || continue
        STEM=${FILE%.raw_data.json}
        NAME=$(basename $STEM)
        DATASET=${NAME%%_*}
        mkdir -p $OUT/$BASEMODEL

        # Likelihood / LogRank / Entropy
        python other_models/baselines.py --dataset_file=$STEM --output_file=$OUT/$BASEMODEL/$NAME \
            --scoring_model_name=gpt-neo-2.7B --cache_dir=./models --seed=42 --dataset=$DATASET
        # DNA-GPT
        python other_models/dna_gpt.py --dataset_file=$STEM --output_file=$OUT/$BASEMODEL/$NAME \
            --base_model_name=gpt-neo-2.7B --cache_dir=./models --seed=42 --dataset=$DATASET
        # DetectGPT (T5-3B perturbations, 100x)
        python other_models/detect_gpt.py --dataset_file=$STEM --output_file=$OUT/$BASEMODEL/$NAME \
            --scoring_model_name=gpt-neo-2.7B --mask_filling_model_name=t5-3b --cache_dir=./models --seed=42 --dataset=$DATASET
        # LRR / NPR (re-use the DetectGPT perturbations)
        python other_models/detect_llm.py --dataset_file=$STEM.t5-3b.perturbation_100 --output_file=$OUT/$BASEMODEL/$NAME \
            --scoring_model_name=gpt-neo-2.7B --cache_dir=./models --seed=42 --dataset=$DATASET
        # Fast-DetectGPT (GPT-J sampling / GPT-Neo-2.7B scoring)
        python other_models/fast_detect_gpt.py --dataset_file=$STEM --output_file=$OUT/$BASEMODEL/$NAME \
            --scoring_model_name=gpt-neo-2.7B --reference_model_name=gpt-j-6B --cache_dir=./models --seed=42 --dataset=$DATASET
    done
done
python tools/show_results.py --result_path $OUT
