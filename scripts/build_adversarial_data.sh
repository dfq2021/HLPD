#!/bin/bash
# Build the adversarial multi-task revision benchmark with API LLMs (Appendix B.3).
#   export HLPD_API_BASE=https://api.openai.com/v1   HLPD_API_KEY=sk-...
MODEL=${MODEL:-gpt-4o}

# Stage 1 - five-dimensional prompt pool (a reasoning model writes the prompts)
python tools/adversarial_prompt_generator.py --n 750 --model deepseek-reasoner \
    --output data/prompts/adversarial_prompts_new.json

# Stage 2 - revise human passages with the target LLM under prompts drawn from the pool
for DATASET in xsum writing squad wiki; do
    python tools/build_revision_data.py --task adversarial \
        --input data/raw/$DATASET.raw_data.json \
        --prompts data/prompts/adversarial_prompts.json \
        --model $MODEL \
        --output data/adversarial_multitask/$MODEL/${DATASET}_adversarial_$MODEL.raw_data.json
done

# Machine-generated text from advanced LLMs (Appendix B.1)
python tools/build_generation_data.py --input data/raw/xsum.raw_data.json --model $MODEL \
    --output data/advanced_generation/$MODEL/xsum_generation_$MODEL.raw_data.json
