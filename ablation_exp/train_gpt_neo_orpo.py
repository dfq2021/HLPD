# Adapted from ImBD (https://github.com/Jiaqi-Chen-00/ImBD, Apache-2.0) for the HLPD ablations
# (Table 4): the preference direction is configurable (`--prefer human` = HLPD setting,
# `--prefer machine` = ImBD setting) and the training file defaults to the HLPD multi-task set.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import torch
import time
import os
import random
import json
import numpy as np
import argparse
import torch
from datasets import Dataset
from trl import ORPOConfig, ORPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, model_name.split("/")[1])
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir,device_map='auto')

model_fullnames = {  
                    'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                    }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_dataset', type=str, default='data/train/train_multitask_revision.raw_data.json')
    parser.add_argument('--prefer', type=str, default='human', choices=['human', 'machine'],
                        help='which side of a pair is preferred (human = HLPD, machine = ImBD)')
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args()
    
    set_seed(args.seed)
    DEVICE = "cuda"
    accumulation_steps = 1
    model_name = "EleutherAI/gpt-neo-2.7B"
    model_save_path="./ckpt/"
    lr=1e-4
    epochs=2
    pad_length = 300
    
    with open(args.train_dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    machine = data.get('rewritten') or data.get('sampled')
    human = data['original']

    model = load_model(model_name, DEVICE,"./models")
    tokenizer = load_tokenizer(model_name,"./models")

    orpodataset = {
        "prompt": ["" for _ in range(len(data['original']))] ,
        "chosen": human if args.prefer == "human" else machine,
        "rejected": machine if args.prefer == "human" else human,
    }
    dataset = Dataset.from_dict(orpodataset)
    
    config = ORPOConfig(
        beta=0.1,
        num_train_epochs = 2,
        per_gpu_train_batch_size=1,
        output_dir="./logs/orpo"
    )
    lora_config =  LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

    orpo_trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
        peft_config=lora_config
    )
    
    orpo_trainer.train()
    model_save_path = os.path.join(model_save_path, args.save_name or f"hlpd_orpo_{args.prefer}")
    if not os.path.exists(model_save_path): os.makedirs(model_save_path)
    # Save the LoRA-modified model
    orpo_trainer.save_model(model_save_path)

    # Save the tokenizer
    tokenizer.save_pretrained(model_save_path)


