"""Humanization adaptive attack (paper Sec. 3.5 / Appendix C).

For every machine-revised passage x_m:
  1. Perturbation generation -- sample ``--n_perturbations`` variants
     ``~x_m ~ q_phi(~x | x_m)`` with a T5 mask-filling model (span masking,
     as in DetectGPT);
  2. Selection -- keep the candidate with the highest sequence
     log-probability under the HLPO-aligned scorer ``p_theta`` (i.e. the
     most human-looking one);
  3. Replacement -- use it as the new x_m and repeat for ``--iterations``.

    python tools/humanize.py \
        --input data/single_task/polish/gpt-3.5-turbo/xsum_polish_gpt-3.5-turbo.raw_data.json \
        --checkpoint models/HLPD-inference --iterations 4 --n 100 \
        --output results/humanize/xsum_polish_gpt-3.5-turbo.humanized.json

The output keeps the original machine text, the humanized text after every
iteration and the HLPD score of each, so that detectors (GPTZero, ImBD, ...)
can be evaluated on the humanized versions.  Use responsibly: this tool is
released to study detector robustness.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "other_models"))

from hlpd.model import load_inference_model, score_text  # noqa: E402
from detect_gpt import load_mask_model, load_mask_tokenizer, perturb_texts  # noqa: E402


@torch.no_grad()
def sequence_logprob(model, tokenizer, text, device):
    tok = tokenizer(text, return_tensors="pt", truncation=True, return_token_type_ids=False).to(device)
    out = model(**tok, labels=tok.input_ids)
    n_tokens = tok.input_ids.shape[1] - 1
    return -out.loss.item() * n_tokens   # sum of token log-probs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="paired data file; the machine texts are attacked")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="./models/HLPD-inference")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--mask_filling_model_name", type=str, default="t5-3b")
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--n", type=int, default=100, help="number of passages")
    parser.add_argument("--n_perturbations", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--pct_words_masked", type=float, default=0.3)
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--mask_top_p", type=float, default=1.0)
    parser.add_argument("--selection", type=str, default="logprob", choices=["logprob", "score"],
                        help="pick the candidate with the highest log-prob (paper) or the lowest HLPD score")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    machine_texts = (data.get("rewritten") or data.get("sampled"))[: args.n]
    originals = data["original"][: len(machine_texts)]

    print("Loading HLPD scorer ...")
    model, tokenizer = load_inference_model(args.checkpoint, base_model=args.base_model,
                                            cache_dir=args.cache_dir, device=args.device)
    print("Loading mask-filling model ...")
    mask_model = load_mask_model(args.mask_filling_model_name, args.device, args.cache_dir)
    mask_model.eval()
    n_positions = getattr(mask_model.config, "n_positions", 512)
    mask_tokenizer = load_mask_tokenizer(args.mask_filling_model_name, n_positions, args.cache_dir)
    perturb_args = SimpleNamespace(span_length=args.span_length, pct_words_masked=args.pct_words_masked,
                                   mask_top_p=args.mask_top_p, device=args.device)

    results = []
    start = time.time()
    for idx, text in enumerate(tqdm.tqdm(machine_texts, desc="Humanizing")):
        record = {"original": originals[idx], "machine": text,
                  "machine_score": score_text(model, tokenizer, text, args.device), "iterations": []}
        current = text
        for it in range(args.iterations):
            candidates = perturb_texts(perturb_args, mask_model, mask_tokenizer, [current] * args.n_perturbations)
            candidates = [c for c in candidates if c.strip()] or [current]
            if args.selection == "logprob":
                values = [sequence_logprob(model, tokenizer, c, args.device) for c in candidates]
                best = int(np.argmax(values))
            else:
                values = [score_text(model, tokenizer, c, args.device) for c in candidates]
                best = int(np.argmin(values))
            current = candidates[best]
            record["iterations"].append({"iteration": it + 1, "text": current,
                                         "hlpd_score": score_text(model, tokenizer, current, args.device),
                                         "selection_value": float(values[best])})
        results.append(record)
        if (idx + 1) % 5 == 0:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Total time: {time.time() - start:.0f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # paired files for each iteration so that any detector can be evaluated on them
    stem = args.output[:-5] if args.output.endswith(".json") else args.output
    for it in range(args.iterations):
        pairs = {"original": [r["original"] for r in results],
                 "rewritten": [r["iterations"][it]["text"] for r in results]}
        with open(f"{stem}.iter{it + 1}.raw_data.json", "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
    mean0 = np.mean([r["machine_score"] for r in results])
    for it in range(args.iterations):
        m = np.mean([r["iterations"][it]["hlpd_score"] for r in results])
        print(f"iteration {it + 1}: mean HLPD score {m:+.3f} (machine-revised input: {mean0:+.3f})")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
