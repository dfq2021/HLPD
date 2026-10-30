#!/bin/bash
# Re-generate the single-task benchmark with open-source instruction models (ImBD protocol).
# Download the generators first (see the commented lines in scripts/download_other_models.sh).
for TASK in rewrite polish expand generation; do
    DATASETS="xsum writing squad"
    if [ "$TASK" = expand ] || [ "$TASK" = generation ]; then DATASETS="xsum"; fi
    for MODELNAME in Qwen2-7B Mistral-7B Llama-3-8B Deepseek-7b; do
        for DATASET in $DATASETS; do
            python tools/data_builder.py \
                --model_name=$MODELNAME --dataset=$DATASET \
                --dataset_file=./data/raw/$DATASET \
                --result_dir=./data/single_task/$TASK/$MODELNAME \
                --task=$TASK
        done
    done
done
