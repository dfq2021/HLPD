"""Repeat HLPO training with several random seeds and report mean AUROC with a
95% confidence interval (Appendix A / Fig. 4-5 of the paper).

Example
-------
python -m hlpd.train_multiseed --seeds 42 199 410 2231 2533 \
    --eval_datasets data/adversarial_multitask/gemini-2.5-pro/*_adversarial_gemini-2.5-pro.raw_data.json \
    --output_dir results/multi_seed/gemini-2.5-pro
"""
import argparse
import gc
import json
import os

import torch
from torch.utils.data import Subset

from .dataset import PairedTextDataset, PairedTextSplitDataset
from .engine import dataset_tag, evaluate, save_result, train
from .metrics import mean_confidence_interval
from .train import build_model, build_parser, set_seed


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 199, 410, 2231, 2533])
    parser.add_argument("--eval_datasets", type=str, nargs="+", default=None,
                        help="one or more evaluation files (default: --eval_dataset)")
    parser.add_argument("--summary_file", type=str, default=None)
    args = parser.parse_args(argv)
    eval_paths = args.eval_datasets or [args.eval_dataset]
    summary_file = args.summary_file or os.path.join(args.output_dir, "multi_seed_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)

    runs = []
    for seed in args.seeds:
        print(f"\n{'=' * 20} seed {seed} {'=' * 20}")
        set_seed(seed)
        args.seed = seed
        gc.collect()
        torch.cuda.empty_cache()
        model = build_model(args)

        train_data = PairedTextSplitDataset(args.train_dataset, split="train", val_ratio=args.val_ratio)
        n = len(train_data) if args.datanum < 0 else min(args.datanum, len(train_data))
        subset = Subset(train_data, torch.randperm(len(train_data))[:n])
        ckpt_dir = os.path.join(args.ckpt_dir or "./ckpt", f"{args.task_name}_seed{seed}")
        run_dir = os.path.join(args.output_dir, f"seed{seed}")

        # train on the first eval set for monitoring, then evaluate on all of them
        first_val = PairedTextDataset(eval_paths[0])
        args.eval_dataset = eval_paths[0]
        model, _ = train(model, subset, first_val, args, device=args.device, ckpt_dir=ckpt_dir)

        entry = {"seed": seed, "checkpoint": ckpt_dir, "metrics": {}}
        for path in eval_paths:
            res = evaluate(model, PairedTextDataset(path), args.device, desc=f"seed {seed} | {dataset_tag(path)}")
            save_result(res, run_dir, path, extra={"seed": seed, "checkpoint": ckpt_dir})
            entry["metrics"][dataset_tag(path)] = {"roc_auc": res["val_ROC_AUC"], "pr_auc": res["val_PR_AUC"]}
        runs.append(entry)
        with open(summary_file, "w") as f:
            json.dump({"runs": runs}, f, indent=2)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = {"runs": runs, "aggregate": {}}
    for path in eval_paths:
        tag = dataset_tag(path)
        aucs = [r["metrics"][tag]["roc_auc"] for r in runs]
        mean, ci = mean_confidence_interval(aucs)
        summary["aggregate"][tag] = {"roc_auc_mean": mean, "roc_auc_ci95": ci, "roc_auc_runs": aucs}
        print(f"{tag}: AUROC = {mean:.4f} +/- {ci:.4f} (95% CI, n={len(aucs)})")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
