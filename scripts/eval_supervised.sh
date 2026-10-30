#!/bin/bash
# Supervised RoBERTa detectors on one benchmark.  Usage: bash scripts/eval_supervised.sh <benchmark>
BENCH=${1:-rewrite}
case $BENCH in
    adversarial_multitask|advanced_generation) DIR=data/$BENCH ;;
    *) DIR=data/single_task/$BENCH ;;
esac
OUT=other_model_results/$BENCH
for MODEL in roberta-base-openai-detector roberta-large-openai-detector; do
    for FILE in $DIR/*/*.raw_data.json; do
        STEM=${FILE%.raw_data.json}
        NAME=$(basename $STEM)
        BASEMODEL=$(basename $(dirname $FILE))
        DATASET=${NAME%%_*}
        mkdir -p $OUT/$BASEMODEL
        python other_models/supervised.py --dataset_file=$STEM --output_file=$OUT/$BASEMODEL/$NAME \
            --model_name=$MODEL --dataset=$DATASET --cache_dir=./models --seed=42
    done
done
