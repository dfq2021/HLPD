# Copied from ImBD (https://github.com/Jiaqi-Chen-00/ImBD, Apache-2.0): builds the single-task
# benchmark (rewrite / polish / expand / generation) with open-source instruction models.
# Prompt pools are read from data/prompts/.
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import argparse
import os
import numpy as np
import random
import json
import tqdm
import time
import re

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

model_fullnames = {  
                     'Mistral-7B': 'mistralai/Mistral-7B-Instruct-v0.3',
                     'Qwen2-7B': 'Qwen/Qwen2-7B-Instruct',
                     'Llama-3-8B': 'meta-llama/Meta-Llama-3-8B-Instruct',
                     'Deepseek-7b': 'deepseek-ai/deepseek-llm-7b-chat',
                    }


def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def from_pretrained(cls, model_name, kwargs, cache_dir="./models"):
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[1])
    else:
        local_path = os.path.join(cache_dir, model_name)
    print(local_path)
    print(os.path.exists(local_path))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, cache_dir=args.cache_dir, trust_remote_code=True, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, trust_remote_code=True, device_map='auto')

def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = dict(torch_dtype=torch.bfloat16)
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model = model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}

    if for_dataset in ['pubmed'] or "Mistral" in model_fullname: # A decoder-only architecture is being used
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
        
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_tokenizer

def llama_filter(messages, generation_args, pipe, output):
    attempt = 0
    while "Can I help you with something else?" in output or output.startswith("I cannot"):
        attempt += 1
        print(f"Can not generate content... Retrying [Attempt {attempt}]")
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text']
        if attempt==15:
            print("Failed to rewrite")
            break
    if output.startswith("Here is a") or output.startswith("Here's a"):
        output = re.sub(r'Here.*?:', '', output, count=1)
    output = output.replace("\n\n","")
    return output

def generate_response_rewrite(original_texts, model, tokenizer, args):
    
    response =[]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.cuda.current_device()
    )
    
    if "Llama" in args.model_name: 
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = pipe.tokenizer.eos_token_id
        
    generation_args = {
        "max_new_tokens": 300,
        "return_full_text": False,
        "eos_token_id": terminators,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
    }
    
    for original_text in tqdm.tqdm(original_texts):
        if "Mistral" in args.model_name or "Deepseek" in args.model_name: # not implementation of system prompt
            messages = [
            {"role": "user", "content": f"You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Original text:{original_text}"},
            {"role": "assistant", "content": f"Here is the rewritten paragraph: "},
        ]
        else:
            messages = [
                {"role": "system", "content": "You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text."},
                {"role": "user", "content": f"{original_text}"},
            ]
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text'].strip()
        if "Llama" in args.model_name:
            output = llama_filter(messages, generation_args, pipe, output)
        response.append(output)
        print(output)    
    return response

def generate_response_polish(original_texts, model, tokenizer, args):
    with open("./data/prompts/polish_prompts_imbd.json","r") as p:
        prompts = json.load(p)['out_prompt']
    response =[]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.cuda.current_device()
    )
    
    if "Llama" in args.model_name: 
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = pipe.tokenizer.eos_token_id
        
    generation_args = {
        "max_new_tokens": 300,
        "return_full_text": False,
        "eos_token_id": terminators,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
    }
    
    for idx, original_text in enumerate(tqdm.tqdm(original_texts)):
        prompt = prompts[idx].strip()
        messages = [
            {"role": "user", "content": f"{prompt}\n{original_text}"},
            {"role": "assistant", "content": f"Here is the polished paragraph: "},
        ]
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text'].strip()
        if "Llama" in args.model_name:
            output = llama_filter(messages, generation_args, pipe, output)
        response.append(output.replace("\n\n"," "))
        print(output)    
    return response

def generate_response_expand(original_texts, model, tokenizer, args):
    with open("./data/prompts/expand_prompts_imbd.json","r") as p:
        prompts = json.load(p)['prompt']
    response =[]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.cuda.current_device()
    )
    
    if "Llama" in args.model_name: 
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = pipe.tokenizer.eos_token_id
        
    generation_args = {
        "max_new_tokens": 300,
        "return_full_text": False,
        "eos_token_id": terminators,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
    }
    
    for idx, original_text in enumerate(tqdm.tqdm(original_texts)):
        prompt = prompts[idx]
        messages = [
            {"role": "user", "content": f"{prompt}\n{original_text}"},
            {"role": "assistant", "content": "Here is the expanded paragraph: "}
        ]
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text'].strip()
        if "Llama" in args.model_name:
            output = llama_filter(messages, generation_args, pipe, output)
        response.append(output)
        print(output)    
    return response

def generate_response_generation(original_texts, model, tokenizer, args):
    response =[]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.cuda.current_device()
    )
    
    if "Llama" in args.model_name: 
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = pipe.tokenizer.eos_token_id
        
    generation_args = {
        "max_new_tokens": 300,
        "return_full_text": False,
        "eos_token_id": terminators,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
    }
    
    for idx, original_text in enumerate(tqdm.tqdm(original_texts)):
        tokenized = tokenizer(original_text, return_tensors='pt',padding=False)
        input_ids = tokenized['input_ids'][0]
        if len(input_ids)>30:
            input_ids = input_ids[:30]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(text)
        
        if args.dataset == "xsum":
            dataset_prompt = "News"
        elif args.dataset == "writing":
            dataset_prompt = "Fiction"
        else:
            dataset_prompt = "Technical"
            
        if "Mistral" in args.model_name or "Deepseek" in args.model_name: # not implementation of system prompt
            messages = [
            {"role": "user", "content": f"You are a {dataset_prompt} writer. Please write an article with about 150 words starting exactly with: {text}"},
        ]
        else:
            messages = [
                {"role": "system", "content": f"You are a {dataset_prompt} writer."},
                {"role": "user", "content": f"Please write an article with about 150 words starting exactly with: {text}"},
            ]
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text'].strip()
        if "Llama" in args.model_name:
            output = llama_filter(messages, generation_args, pipe, output)
        response.append(output)
        print(output)    
    return response


def forward(args):

    with open(args.dataset_file + ".raw_data.json","r") as f:
        original_texts = json.load(f)['original']
    model = load_model(args.model_name, device="cuda",cache_dir=args.cache_dir)
    tokenizer = load_tokenizer(args.model_name, for_dataset=args.dataset, cache_dir=args.cache_dir)
    if args.task == "rewrite":
        texts = generate_response_rewrite(original_texts, model, tokenizer, args)
    elif args.task == "polish":
        texts = generate_response_polish(original_texts, model, tokenizer, args)
    elif args.task == "expand":
        texts = generate_response_expand(original_texts, model, tokenizer, args)
    else:
        texts = generate_response_generation(original_texts, model, tokenizer, args)
        
    data = dict(original=original_texts ,rewritten=texts)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    with open(os.path.join(args.result_dir, f"{args.dataset}_{args.task}_{args.model_name}.raw_data.json"), 'w') as fout:
        json.dump(data, fout)
        print(f'Data written into {args.result_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default="Llama-3-8B")
    parser.add_argument('--dataset_file', type=str, default="./data/xsum")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--result_dir', type=str, default="./data/polish/gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="./models")
    parser.add_argument('--task', type=str, choices=["rewrite", "polish", "expand", "generation"], default="polish")
    args = parser.parse_args()
    
    set_seed(args.seed)
    forward(args)
    
    