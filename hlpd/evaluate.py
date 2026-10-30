"""Fast evaluation with an inference checkpoint (base LM + LoRA adapter only,
~11 GB GPU memory).  The model is loaded once and every ``--eval_dataset``
file (globs allowed) is scored with the HLP-CPC criterion.

Example
-------
python -m hlpd.evaluate --checkpoint models/HLPD-inference \
    --eval_dataset "data/adversarial_multitask/*/*.raw_data.json" \
    --output_dir evaluation/adversarial_multitask
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import tqdm

from .dataset import load_pairs
from .engine import dataset_tag, save_result
from .metrics import get_precision_recall_metrics, get_roc_metrics
from .model import load_inference_model, score_text


@torch.no_grad()
def evaluate_file(model, tokenizer, path, device):
    human, machine = load_pairs(path)
    crit_real, crit_sampled = [], []
    start = time.time()
    for h, m in tqdm.tqdm(list(zip(human, machine)), desc=f"Evaluating {dataset_tag(path)}"):
        crit_real.append(score_text(model, tokenizer, h, device))
        crit_sampled.append(score_text(model, tokenizer, m, device))
    elapsed = time.time() - start
    fpr, tpr, roc_auc = get_roc_metrics(crit_real, crit_sampled)
    p, r, pr_auc = get_precision_recall_metrics(crit_real, crit_sampled)
    print(f"{dataset_tag(path)}: ROC AUC = {roc_auc:.4f}  PR AUC = {pr_auc:.4f}  ({elapsed:.1f}s)")
    return {
        "name": "HLPD",
        "val_ROC_AUC": roc_auc, "val_PR_AUC": pr_auc,
        "val_crit_real_mean": float(np.mean(crit_real)), "val_crit_real_std": float(np.std(crit_real)),
        "val_crit_sampled_mean": float(np.mean(crit_sampled)), "val_crit_sampled_std": float(np.std(crit_sampled)),
        "n_pairs": len(human), "time_seconds": elapsed,
        "crit_real": crit_real, "crit_sampled": crit_sampled,
        "fpr": fpr, "tpr": tpr, "precision": p, "recall": r,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="HLPD fast evaluation")
    parser.add_argument("--checkpoint", type=str, default="./models/HLPD-inference",
                        help="directory with adapter_model.safetensors")
    parser.add_argument("--base_model", type=str, default=None, help="override base model recorded in the adapter")
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--eval_dataset", type=str, nargs="+", required=True, help="data files or globs")
    parser.add_argument("--output_dir", type=str, default="./evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", help="load the model in float16")
    args = parser.parse_args(argv)

    files = []
    for pattern in args.eval_dataset:
        matched = sorted(glob.glob(pattern))
        files.extend(matched if matched else [pattern])
    print(f"{len(files)} evaluation file(s)")

    print("Loading model ...")
    start = time.time()
    model, tokenizer = load_inference_model(args.checkpoint, base_model=args.base_model, cache_dir=args.cache_dir,
                                            device=args.device, torch_dtype=torch.float16 if args.fp16 else None)
    print(f"Done ({time.time() - start:.1f}s)")

    summary = []
    for path in files:
        res = evaluate_file(model, tokenizer, path, args.device)
        save_result(res, args.output_dir, path, extra={"checkpoint": args.checkpoint})
        summary.append((dataset_tag(path), res["val_ROC_AUC"]))
    print("\n== Summary (AUROC) ==")
    for tag, auc in summary:
        print(f"{auc:.4f}  {tag}")


if __name__ == "__main__":
    main()
