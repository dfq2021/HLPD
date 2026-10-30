"""Evaluate the commercial GPTZero detector (API) on a paired data file.

    export GPTZERO_API_KEY=...
    python other_models/gptzero.py \
        --dataset_file data/adversarial_multitask/gpt-4o/xsum_adversarial_gpt-4o \
        --output_file other_model_results/adversarial_multitask/gpt-4o/xsum_adversarial_gpt-4o

The output follows the format of the other baselines (``metrics.roc_auc``)
so that ``tools/show_results.py`` can read it.  Requests are cached in the
output file, so an interrupted run can be resumed.
"""
import argparse
import json
import os
import time

import numpy as np
import requests
import tqdm

from metrics import get_precision_recall_metrics, get_roc_metrics

API_URL = "https://api.gptzero.me/v2/predict/text"


def gptzero_prob(text, api_key, version="2024-07-12-base", retries=5):
    headers = {"Accept": "application/json", "Content-Type": "application/json", "x-api-key": api_key}
    payload = {"document": text, "version": version, "multilingual": False}
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return float(r.json()["documents"][0]["completely_generated_prob"])
        except Exception as e:  # noqa: BLE001
            wait = 5 * (attempt + 1)
            print(f"GPTZero request failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    return float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="path without the .raw_data.json suffix")
    parser.add_argument("--output_file", type=str, required=True, help="path without the .gptzero.json suffix")
    parser.add_argument("--api_key", type=str, default=os.environ.get("GPTZERO_API_KEY"))
    parser.add_argument("--version", type=str, default="2024-07-12-base")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not args.api_key:
        raise SystemExit("set GPTZERO_API_KEY or pass --api_key")

    with open(f"{args.dataset_file}.raw_data.json", encoding="utf-8") as f:
        data = json.load(f)
    machine = data.get("rewritten") or data.get("sampled")
    n = min(len(data["original"]), len(machine))

    results_file = f"{args.output_file}.gptzero.json"
    raw = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            raw = json.load(f).get("raw_results", [])
        print(f"resuming from {len(raw)} scored pairs")

    for idx in tqdm.tqdm(range(len(raw), n), desc="GPTZero"):
        raw.append({"original": data["original"][idx],
                    "original_crit": gptzero_prob(data["original"][idx], args.api_key, args.version),
                    "rewritten": machine[idx],
                    "rewritten_crit": gptzero_prob(machine[idx], args.api_key, args.version)})
        time.sleep(args.delay)
        if idx % 10 == 0:
            os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
            with open(results_file, "w") as f:
                json.dump({"name": "gptzero", "raw_results": raw}, f)

    real = [x["original_crit"] for x in raw if not np.isnan(x["original_crit"])]
    samples = [x["rewritten_crit"] for x in raw if not np.isnan(x["rewritten_crit"])]
    fpr, tpr, roc_auc = get_roc_metrics(real, samples)
    p, r, pr_auc = get_precision_recall_metrics(real, samples)
    print(f"GPTZero ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump({"name": "gptzero_threshold", "info": {"n_samples": n, "version": args.version},
                   "predictions": {"real": real, "samples": samples}, "raw_results": raw,
                   "metrics": {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
                   "pr_metrics": {"pr_auc": pr_auc, "precision": p, "recall": r}, "loss": 1 - pr_auc}, f)
    print(f"Results written into {results_file}")


if __name__ == "__main__":
    main()
