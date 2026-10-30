# Adapted from ImBD (https://github.com/Jiaqi-Chen-00/ImBD, Apache-2.0) for the HLPD ablations
# (Table 4): the preference direction is configurable (`--prefer human` = HLPD setting,
# `--prefer machine` = ImBD setting) and the training file defaults to the HLPD multi-task set.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AdamW
from torch.cuda.amp import GradScaler, autocast
import time
from torch import nn
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
import os
import random
import json
import numpy as np
import argparse

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
    parser.add_argument('--datanum', type=int, default=1000)
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
    model = load_model(model_name,DEVICE,"./models")
    tokenizer = load_tokenizer(model_name,"./models")
    pad_length = 250
    datanum = args.datanum
    
    with open(args.train_dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    # SFT on the preferred side only (human text for HLPD, machine text for ImBD)
    train_set = data['original'] if args.prefer == 'human' else (data.get('rewritten') or data.get('sampled'))
    print(len(train_set))
    processed_data = []
    max_len=0
    for sentence in train_set:
        tokenized = tokenizer(sentence, return_tensors='pt',padding=False)
        input_ids = tokenized['input_ids'][0]
        max_len = max(len(input_ids),max_len)
        if len(input_ids) > 30:
            input_tokens = input_ids[:30]
            label_tokens = input_ids[30:]
            input_tokens_padded = F.pad(input_tokens, (0, pad_length - len(input_tokens)), value=tokenizer.pad_token_id)
            label_tokens_padded = F.pad(label_tokens, (0, pad_length - len(label_tokens)), value=tokenizer.pad_token_id)
            processed_data.append({
                "input": input_tokens_padded,
                "label": label_tokens_padded
            })
    if len(processed_data) > datanum:
        processed_data = random.sample(processed_data, datanum)      
    
    lora_config =  LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

    model = get_peft_model(model, lora_config)

    epoch_losses, i, loss = [], 0, torch.tensor(0.0).to(DEVICE)
    epoch_crit_train_original, epoch_crit_train_rewritten = [],[]
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(processed_data) * epochs, eta_min=0,
                                  last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()
    model.to(DEVICE)
    print(len(processed_data))
    for epoch in range(epochs):
        optimizer.zero_grad()
        for item in tqdm.tqdm(processed_data, desc=f"Fine-tuning: {epoch} epoch"):
            inp, target = item["input"].to(DEVICE), item["label"].to(DEVICE)
            scheduler.step()
            with autocast():
                outputs = model(input_ids=inp.unsqueeze(0),labels=target.unsqueeze(0))
                loss += outputs.loss
            
            if ((i + 1) % accumulation_steps) == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                epoch_losses.append(loss.item())
                loss = torch.tensor(0.0).to(DEVICE)
                
            i += 1
        print(np.mean(epoch_losses))
        epoch_losses=[]
    save_name = args.save_name or f"hlpd_sft_{args.prefer}_{datanum}"
    model.save_pretrained(os.path.join(model_save_path, save_name))
    tokenizer.save_pretrained(os.path.join(model_save_path, save_name))
