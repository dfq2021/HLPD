"""Interactive command-line detector (local demo).

    python -m hlpd.detect                          # interactive
    python -m hlpd.detect --text "some passage"    # one-shot
    python -m hlpd.detect --file passages.txt      # one passage per blank-line separated block
    python -m hlpd.detect --file passages.json     # {"texts": [...]} or a JSON list

The scoring model is the HLPO-aligned GPT-Neo-2.7B (LoRA adapter in
``--checkpoint``).  Scores are converted to probabilities with the reference
score distributions in ``--ref_path`` (see ``prob_estimator.py``).
"""
import argparse
import json
import os
import sys
import time

import torch

from . import __version__
from .model import load_inference_model, score_text
from .prob_estimator import SCENARIO_LABELS, SCENARIOS, ProbEstimator

BANNER = r"""
  _   _  _     ____   ____
 | | | || |   |  _ \ |  _ \    Human Language Preference Detection
 | |_| || |   | |_) || | | |   machine-revised & machine-generated text detector
 |  _  || |___|  __/ | |_| |   AAAI 2026  |  v{version}
 |_| |_||_____|_|    |____/
"""

USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("HLPD_COLOR") == "1"


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bar(prob, width=30):
    filled = int(round(prob * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def verdict(prob):
    if prob >= 0.8:
        return c("very likely MACHINE-revised", "91")
    if prob >= 0.6:
        return c("likely machine-revised", "93")
    if prob > 0.4:
        return c("uncertain", "33")
    if prob > 0.2:
        return c("likely human-written", "92")
    return c("very likely HUMAN-written", "92")


def read_texts_from_file(path):
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("texts") or data.get("original") or list(data.values())[0]
        return [str(t) for t in data]
    with open(path, encoding="utf-8") as f:
        blocks = [b.strip() for b in f.read().split("\n\n")]
    return [b for b in blocks if b]


def read_interactive():
    print(c("Please enter your text (finish with an empty line, or type 'q' to quit):", "36"))
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            return None
        if line.strip().lower() == "q" and not lines:
            return None
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def report(text, score, estimator, scenario, detail):
    n_words = len(text.split())
    prob = estimator.prob(score, scenario)
    label = "machine-revised/generated" if scenario == "all" else SCENARIO_LABELS[scenario]
    print()
    print(f"  HLPD score (-HLP-CPC) : {score:+.4f}")
    print(f"  P({label}) : {prob * 100:5.1f}%  {bar(prob)}")
    print(f"  Verdict               : {verdict(prob)}")
    if detail:
        print("  Per-scenario probability:")
        for s, p in estimator.prob_detail(score).items():
            print(f"    {s:<12s} {p * 100:5.1f}%  {bar(p, 20)}")
    if n_words < 50:
        print(c(f"  ! only {n_words} words -- results on very short passages are unreliable", "33"))
    print()
    return prob


def main(argv=None):
    parser = argparse.ArgumentParser(description="HLPD local detector")
    parser.add_argument("--checkpoint", type=str, default="./models/HLPD-inference")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--ref_path", type=str, default="./local_infer_ref")
    parser.add_argument("--scenario", type=str, default="all", choices=SCENARIOS + ["all"],
                        help="reference distribution used to calibrate the probability")
    parser.add_argument("--detail", action="store_true", help="show per-scenario probabilities")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--text", type=str, default=None, help="score a single passage and exit")
    parser.add_argument("--file", type=str, default=None, help="score every passage in a .txt/.json file")
    parser.add_argument("--output", type=str, default=None, help="write batch results as JSON")
    args = parser.parse_args(argv)

    print(BANNER.format(version=__version__))
    print("Loading model ...", end=" ", flush=True)
    start = time.time()
    model, tokenizer = load_inference_model(args.checkpoint, base_model=args.base_model, cache_dir=args.cache_dir,
                                            device=args.device, torch_dtype=torch.float16 if args.fp16 else None)
    print(f"done ({time.time() - start:.1f}s, device={args.device})")
    estimator = ProbEstimator(args.ref_path)
    if args.scenario != "all" and args.scenario not in estimator.available:
        print(f"no reference scores for scenario '{args.scenario}', falling back to 'all'")
        args.scenario = "all"

    if args.text or args.file:
        texts = [args.text] if args.text else read_texts_from_file(args.file)
        results = []
        for i, t in enumerate(texts):
            score = score_text(model, tokenizer, t, args.device)
            print(c(f"--- passage {i + 1}/{len(texts)} ({len(t.split())} words) ---", "36"))
            prob = report(t, score, estimator, args.scenario, args.detail)
            results.append({"index": i, "score": score, "prob_machine": prob,
                            "detail": estimator.prob_detail(score) if args.detail else None, "text": t})
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results written to {args.output}")
        return

    print("Local demo for HLPD. Longer passages give more reliable results.")
    print("Use --detail for per-scenario probabilities, --scenario to pick a reference distribution.\n")
    while True:
        text = read_interactive()
        if text is None or not text.strip():
            print("bye.")
            break
        score = score_text(model, tokenizer, text, args.device)
        report(text, score, estimator, args.scenario, args.detail)


if __name__ == "__main__":
    main()
