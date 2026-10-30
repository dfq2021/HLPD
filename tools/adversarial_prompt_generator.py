"""Five-dimensional adversarial prompt generator (paper Appendix B.3).

Stage 1 of the adversarial multi-task revision pipeline.  Each meta-instruction
combines five randomly drawn components

    (1) revision goal  (2) target style  (3) adversarial "sound human" request
    (4) operational constraint  (5) auxiliary request

and asks a strong LLM (DeepSeek-R1 in the paper) to write a compact user
prompt with them.  The returned prompts are cleaned (longest quoted span)
and stored as ``{"prompts": [...]}``; stage 2 (``build_revision_data.py``)
applies them to human-written passages.

The API is any OpenAI-compatible endpoint:

    export HLPD_API_BASE=https://api.deepseek.com/v1
    export HLPD_API_KEY=sk-...
    python tools/adversarial_prompt_generator.py --n 750 --model deepseek-reasoner \
        --output data/prompts/adversarial_prompts_new.json
"""
import argparse
import json
import logging
import os
import random
import re
import time

REVISION_GOALS = ["Paraphrase", "Rewrite", "Polish", "Expand", "Restructure"]
STYLE_CONTROLS = ["formal", "oral", "academic", "literary", "critical", "narrative",
                  "descriptive", "lyric", "objective", "subjective", "technical"]
ADVERSARIAL_TEXT = ["make the paragraph sound as human as possible",
                    "make this paragraph feel more natural, like a real person wrote it",
                    "make this text sound less robotic and more human"]
CONSTRAINTS = ["keep factual accuracy", "no hallucinated content", "maintain original intent"]
ADDITIONAL_OPTS = ["enhance expression", "make sentences more concise", "reorganize the structure",
                   "preserve all factual details", "restructure sentences for better flow"]
WORD_LENS = [45, 50, 55]


def sample_meta_instruction(rng):
    """Draw one component per dimension and build the meta-instruction."""
    parts = dict(
        word_len=rng.choice(WORD_LENS),
        revision_goal=rng.choice(REVISION_GOALS).lower(),
        style=rng.choice(STYLE_CONTROLS),
        adversarial=rng.choice(ADVERSARIAL_TEXT),
        additional_opt=rng.choice(ADDITIONAL_OPTS),
        constraint=rng.choice(CONSTRAINTS),
    )
    text = ("Create a prompt in {word_len} words that says you want GPT's help to {revision_goal} "
            "a paragraph in a {style} style, {adversarial}, {additional_opt}, and {constraint}. "
            "Return only the prompt, wrapped in double quotes.").format(**parts)
    return text, parts


def clean_prompt(raw):
    """Keep the longest double-quoted span (the prompt itself), drop word counts etc."""
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()   # reasoning models
    quoted = re.findall(r'"([^"]{20,})"', raw)
    if quoted:
        return max(quoted, key=len).strip()
    raw = re.sub(r"\(\s*\d+\s*words?\s*\)\s*$", "", raw).strip()
    return raw.strip('"').strip()


def make_client(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def complete(client, model, instruction, temperature=0.7, retries=8):
    from openai import APIConnectionError, RateLimitError
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "user", "content": instruction}])
            return resp.choices[0].message.content.strip()
        except (APIConnectionError, RateLimitError) as e:
            wait = min(60, 5 * 2 ** attempt)
            logging.warning(f"{type(e).__name__}: retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("API call failed repeatedly")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=750, help="number of prompts")
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--api_base", type=str, default=os.environ.get("HLPD_API_BASE"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("HLPD_API_KEY"))
    parser.add_argument("--output", type=str, default="data/prompts/adversarial_prompts_new.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true", help="only print meta-instructions")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rng = random.Random(args.seed)
    if args.dry_run:
        for _ in range(5):
            print(sample_meta_instruction(rng)[0])
        return
    if not args.api_key:
        raise SystemExit("set HLPD_API_KEY (and HLPD_API_BASE) or pass --api_key/--api_base")
    client = make_client(args.api_base, args.api_key)

    results = {"prompts": [], "meta": []}
    if os.path.exists(args.output):  # resume
        with open(args.output, encoding="utf-8") as f:
            results = json.load(f)
        logging.info(f"resuming with {len(results['prompts'])} prompts")
    seen = set(results["prompts"])
    while len(results["prompts"]) < args.n:
        instruction, parts = sample_meta_instruction(rng)
        try:
            prompt = clean_prompt(complete(client, args.model, instruction))
        except Exception as e:
            logging.error(f"failed: {e}")
            continue
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        results["prompts"].append(prompt)
        results["meta"].append(parts)
        logging.info(f"[{len(results['prompts'])}/{args.n}] {prompt[:100]}")
        if len(results["prompts"]) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        time.sleep(args.delay)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"{len(results['prompts'])} prompts saved to {args.output}")


if __name__ == "__main__":
    main()
