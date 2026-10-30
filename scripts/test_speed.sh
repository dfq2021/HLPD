#!/bin/bash
# Inference cost (Table 1, seconds per 1k words) on 10 XSum passages.
python -m hlpd.evaluate --checkpoint models/HLPD-inference \
    --eval_dataset data/truncation/speed/xsum_polish_gpt-3.5-turbo_10.raw_data.json --output_dir evaluation/speed
python tools/count_words.py -i data/truncation/speed/xsum_polish_gpt-3.5-turbo_10.raw_data.json -s 10
