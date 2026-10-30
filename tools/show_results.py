"""Print AUROC of result files and (optionally) a Markdown table.

    python tools/show_results.py --result_path results/paper/adversarial_multitask --table
    python tools/show_results.py --result_path other_model_results/rewrite/gpt-4o --other

Result files are ``*.hlpd.json`` (``val_ROC_AUC``) or baseline outputs of
``other_models/*`` (``metrics.roc_auc``); both are recognised automatically.
"""
import argparse
import glob
import json
import os
from collections import OrderedDict, defaultdict

DATASET_ORDER = ["xsum", "writing", "squad", "pubmed", "wiki"]


def read_auroc(path):
    with open(path) as f:
        d = json.load(f)
    if "val_ROC_AUC" in d:
        return float(d["val_ROC_AUC"])
    if "metrics" in d and "roc_auc" in d["metrics"]:
        return float(d["metrics"]["roc_auc"])
    return None


def dataset_of(filename):
    name = os.path.basename(filename).split(".")[0]
    for ds in DATASET_ORDER:
        if name == ds or name.startswith(ds + "_"):
            return ds
    return name


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="results/paper/adversarial_multitask")
    parser.add_argument("--filter", type=str, default="", help="substring filter on file names")
    parser.add_argument("--other", action="store_true", help="(kept for ImBD compatibility; format is auto-detected)")
    parser.add_argument("--table", action="store_true", help="print a Markdown table: rows = sub-directory, columns = dataset")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.result_path, "**", "*.json"), recursive=True))
    files = [f for f in files if args.filter in os.path.basename(f)]
    rows = defaultdict(OrderedDict)
    for f in files:
        auc = read_auroc(f)
        if auc is None:
            continue
        rel = os.path.relpath(f, args.result_path)
        group = os.path.dirname(rel) or "."
        print(f"{auc:.4f}  {rel}")
        rows[group][dataset_of(f)] = auc

    if args.table and rows:
        datasets = [d for d in DATASET_ORDER if any(d in r for r in rows.values())]
        extra = sorted({d for r in rows.values() for d in r} - set(datasets))
        datasets += extra
        print("\n| source | " + " | ".join(datasets) + " | avg |")
        print("|---|" + "---:|" * (len(datasets) + 1))
        for group, r in rows.items():
            vals = [r.get(d) for d in datasets]
            cells = [f"{v:.4f}" if v is not None else "–" for v in vals]
            present = [v for v in vals if v is not None]
            avg = sum(present) / len(present) if present else float("nan")
            print(f"| {group} | " + " | ".join(cells) + f" | {avg:.4f} |")


if __name__ == "__main__":
    main()
