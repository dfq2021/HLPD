#!/bin/bash
# Commercial GPTZero API on one benchmark (needs GPTZERO_API_KEY).
# Usage: bash scripts/eval_gptzero.sh <benchmark>
BENCH=${1:-adversarial_multitask}
case $BENCH in
    adversarial_multitask|advanced_generation) DIR=data/$BENCH ;;
    *) DIR=data/single_task/$BENCH ;;
esac
for FILE in $DIR/*/*.raw_data.json; do
    STEM=${FILE%.raw_data.json}
    NAME=$(basename $STEM)
    BASEMODEL=$(basename $(dirname $FILE))
    mkdir -p other_model_results/$BENCH/$BASEMODEL
    python other_models/gptzero.py --dataset_file=$STEM --output_file=other_model_results/$BENCH/$BASEMODEL/$NAME
done
