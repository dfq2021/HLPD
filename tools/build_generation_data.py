"""Machine-*generated* text with API LLMs (paper Appendix B.1).

The LLM continues a human-written passage from its first ``--prefix_words``
words, without any stylistic instruction:

    "You are a News writer. Please write an article with about 150 words
     starting exactly with: <prefix>"

    export HLPD_API_BASE=...   HLPD_API_KEY=...
    python tools/build_generation_data.py --input data/raw/xsum.raw_data.json \
        --model deepseek-reasoner \
        --output data/advanced_generation/deepseek-r1/xsum_generation_deepseek-r1.raw_data.json

Output: ``{"original": [...human...], "rewritten": [...generated...]}``.
"""
import argparse
import json
import logging
import os
import re
import time

SYSTEM_PROMPT = "You are a News writer. Please write an article with about {n_words} words starting exactly with: {prefix}"


def make_client(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def complete(client, model, prefix, n_words, temperature=0.7, retries=8):
    from openai import APIConnectionError, RateLimitError
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "system", "content": SYSTEM_PROMPT.format(n_words=n_words, prefix=prefix)},
                          {"role": "user", "content": prefix}])
            return resp.choices[0].message.content.strip()
        except (APIConnectionError, RateLimitError) as e:
            wait = min(60, 5 * 2 ** attempt)
            logging.warning(f"{type(e).__name__}: retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("API call failed repeatedly")


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\*\*(.*?)\*\*", r"\1", text.strip())   # bold headline from some models
    return text.strip()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api_base", type=str, default=os.environ.get("HLPD_API_BASE"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("HLPD_API_KEY"))
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--prefix_words", type=int, default=30)
    parser.add_argument("--target_words", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.api_key:
        raise SystemExit("set HLPD_API_KEY (and HLPD_API_BASE) or pass --api_key/--api_base")

    with open(args.input, encoding="utf-8") as f:
        originals = json.load(f)["original"][: args.n]
    result = {"original": originals, "rewritten": []}
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            prev = json.load(f)
        result["rewritten"] = prev.get("rewritten", [])
        logging.info(f"resuming from {len(result['rewritten'])} passages")

    client = make_client(args.api_base, args.api_key)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for idx in range(len(originals)):
        if idx < len(result["rewritten"]) and result["rewritten"][idx]:
            continue
        prefix = " ".join(originals[idx].split()[: args.prefix_words])
        try:
            text = clean(complete(client, args.model, prefix, args.target_words, args.temperature))
        except Exception as e:
            logging.error(f"passage {idx} failed: {e}")
            text = ""
        if idx < len(result["rewritten"]):
            result["rewritten"][idx] = text
        else:
            result["rewritten"].append(text)
        logging.info(f"[{idx + 1}/{len(originals)}] {text[:80]!r}")
        if (idx + 1) % args.save_every == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        time.sleep(args.delay)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logging.info(f"saved {len(result['rewritten'])} pairs to {args.output}")


if __name__ == "__main__":
    main()
