"""Train (HLPO) or evaluate an HLPD scoring model.

Examples
--------
# HLPO training on the multi-task revision set (paper setting)
python -m hlpd.train --task_name hlpd_multitask --epochs 2 --lr 1e-4 --beta 0.05

# evaluation only, with a trained adapter
python -m hlpd.train --eval_only --from_pretrained ./ckpt/hlpd_multitask_hlpo_lr_0.0001_beta_0.05_a_1 \
    --eval_dataset data/adversarial_multitask/gpt-4o/xsum_adversarial_gpt-4o.raw_data.json \
    --output_dir results/adversarial_multitask/gpt-4o

# ablations of the objective
python -m hlpd.train --loss sigmoid --no_dynamic_beta   # DPO/ImBD-style objective, fixed beta
python -m hlpd.train --loss linear  --no_dynamic_beta   # HLPO‡  (linear, fixed beta)
python -m hlpd.train --loss sigmoid                     # HLPO⋆  (sigmoid, Dynamic-beta)
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Subset

from .dataset import PairedTextDataset, PairedTextSplitDataset
from .engine import run
from .hlpo_loss import HLPOLoss
from .model import HLPDScorer

DEFAULT_TRAIN = "data/train/train_multitask_revision.raw_data.json"
DEFAULT_EVAL = "data/adversarial_multitask/gpt-4o/xsum_adversarial_gpt-4o.raw_data.json"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    p = argparse.ArgumentParser(description="HLPD: HLPO training / HLP-CPC evaluation")
    # data
    p.add_argument("--train_dataset", type=str, default=DEFAULT_TRAIN)
    p.add_argument("--eval_dataset", type=str, default=DEFAULT_EVAL)
    p.add_argument("--datanum", type=int, default=-1, help="number of training pairs (-1 = all)")
    p.add_argument("--val_ratio", type=float, default=0.0, help="hold out part of the training file")
    # optimisation
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("-a", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    # HLPO objective
    p.add_argument("--beta", type=float, default=0.05, help="initial / fixed beta")
    p.add_argument("--loss", type=str, default="linear", choices=["linear", "sigmoid"])
    p.add_argument("--no_dynamic_beta", action="store_true", help="use a fixed beta")
    p.add_argument("--beta_scheduler", type=str, default="ema", choices=["ema", "window"])
    p.add_argument("--beta_max", type=float, default=2.0)
    p.add_argument("--beta_min", type=float, default=0.1)
    p.add_argument("--beta_window", type=int, default=10)
    p.add_argument("--beta_low_var", type=float, default=0.5)
    p.add_argument("--beta_high_var", type=float, default=0.7)
    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--no_lora", action="store_true",
                   help="evaluate a fully fine-tuned base model (SFT / RLHF / ORPO ablations)")
    p.add_argument("--share_reference", action="store_true",
                   help="reference = scoring model with the adapter disabled (one model in memory, fits 24 GB GPUs)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="recompute activations in the backward pass (less GPU memory, ~30%% slower)")
    # models / io
    p.add_argument("--base_model", type=str, default="gpt-neo-2.7B")
    p.add_argument("--cache_dir", type=str, default="./models")
    p.add_argument("--from_pretrained", type=str, default=None, help="adapter (or legacy) checkpoint dir")
    p.add_argument("--task_name", type=str, default="hlpd_multitask")
    p.add_argument("--ckpt_dir", type=str, default=None, help="override checkpoint directory")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_freq", type=int, default=1, help="evaluate every N epochs")
    p.add_argument("--save_every_eval", action="store_true")
    p.add_argument("-ebt", action="store_true", help="evaluate before tuning")
    p.add_argument("--eval_only", action="store_true")
    return p


def build_model(args):
    loss = HLPOLoss(beta=args.beta, loss_type=args.loss, dynamic_beta=not args.no_dynamic_beta,
                    scheduler=args.beta_scheduler, beta_max=args.beta_max, beta_min=args.beta_min,
                    window_size=args.beta_window, low_var=args.beta_low_var, high_var=args.beta_high_var)
    model = HLPDScorer(args.base_model, args.base_model, loss=loss, use_lora=not args.no_lora,
                       device=args.device, cache_dir=args.cache_dir,
                       lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                       share_reference=args.share_reference)
    if args.from_pretrained:
        print(f"Loading checkpoint from {args.from_pretrained} ...")
        model.from_pretrained(args.from_pretrained)
    if args.gradient_checkpointing and not args.eval_only:
        model.scoring_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.scoring_model.config.use_cache = False
    return model


def default_ckpt_dir(args):
    tag = f"{args.task_name}_hlpo_lr_{args.lr}_beta_{args.beta}_a_{args.a}"
    if args.loss != "linear":
        tag += f"_{args.loss}"
    if args.no_dynamic_beta:
        tag += "_fixedbeta"
    return os.path.join("./ckpt", tag)


def main(args=None):
    args = build_parser().parse_args(args)
    set_seed(args.seed)
    print(f"Running with args: {vars(args)}")
    model = build_model(args)

    val_data = PairedTextDataset(args.eval_dataset)
    train_subset = None
    if not args.eval_only:
        train_data = PairedTextSplitDataset(args.train_dataset, split="train", val_ratio=args.val_ratio)
        n = len(train_data) if args.datanum < 0 else min(args.datanum, len(train_data))
        indices = torch.randperm(len(train_data))[:n]
        train_subset = Subset(train_data, indices)
        ckpt_dir = args.ckpt_dir or default_ckpt_dir(args)
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "train_indices.json"), "w") as f:
            json.dump({"train_dataset": args.train_dataset, "indices": indices.tolist(), "seed": args.seed}, f)
        print(f"train pairs: {len(train_subset)}   eval pairs: {len(val_data)}   ckpt: {ckpt_dir}")
    else:
        ckpt_dir = args.ckpt_dir or "./ckpt"
        print(f"eval pairs: {len(val_data)}")

    return run(model, train_subset, val_data, args, device=args.device, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()
