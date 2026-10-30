"""Stage 2 of the adversarial multi-task pipeline: revise human-written
passages with an API LLM under prompts drawn from a prompt pool.

Also usable for the single-task revision protocols of ImBD (``--task
rewrite|polish|expand``) with API models.

    export HLPD_API_BASE=...   HLPD_API_KEY=...
    python tools/build_revision_data.py \
        --input data/raw/xsum.raw_data.json \
        --prompts data/prompts/adversarial_prompts.json \
        --model gpt-4o --model_tag gpt-4o \
        --output data/adversarial_multitask/gpt-4o/xsum_adversarial_gpt-4o.raw_data.json

Output format: ``{"original": [...], "rewritten": [...]}`` (aligned lists).
Progress is saved every ``--save_every`` passages and the script resumes
from an existing output file.
"""
import argparse
import json
import logging
import os
import random
import re
import time

REWRITE_PROMPT = ("You are a professional rewriting expert and you can help paraphrasing this paragraph "
                  "without missing the original details. Please keep the length of the rewritten text "
                  "similar to the original text.")
EXPAND_PROMPT = "Expand but not extend the paragraph in a {style} style."
STYLES = ["formal", "oral", "academic", "literary", "critical", "narrative", "descriptive",
          "lyric", "objective", "subjective"]
WRAP_INSTRUCTION = " Output only the revised paragraph, wrapped between <begin> and <end>."


def strip_markers(text):
    """Remove ``<begin>``/``<end>`` wrappers (and markdown code fences) if present."""
    if "<begin>" in text:
        text = text.split("<begin>", 1)[1]
    if "<end>" in text:
        text = text.split("<end>", 1)[0]
    text = re.sub(r"^```[a-z]*\n?|```$", "", text.strip())
    return text.strip()


def make_client(base_url, api_key):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def complete(client, model, system_prompt, user_text, temperature=0.7, retries=8):
    from openai import APIConnectionError, RateLimitError
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_text}])
            return resp.choices[0].message.content.strip()
        except (APIConnectionError, RateLimitError) as e:
            wait = min(60, 5 * 2 ** attempt)
            logging.warning(f"{type(e).__name__}: retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("API call failed repeatedly")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSON with an 'original' list")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--task", type=str, default="adversarial", choices=["adversarial", "rewrite", "polish", "expand"])
    parser.add_argument("--prompts", type=str, default="data/prompts/adversarial_prompts.json",
                        help="prompt pool ({'prompts': [...]} or {'out_prompt': [...]}) for adversarial / polish")
    parser.add_argument("--prompt_order", type=str, default="cyclic", choices=["cyclic", "random"])
    parser.add_argument("--model", type=str, required=True, help="API model name")
    parser.add_argument("--api_base", type=str, default=os.environ.get("HLPD_API_BASE"))
    parser.add_argument("--api_key", type=str, default=os.environ.get("HLPD_API_KEY"))
    parser.add_argument("--n", type=int, default=150, help="number of passages (from the start of the file)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--wrap", action="store_true", help="ask for <begin>/<end> wrappers (robust parsing)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.api_key:
        raise SystemExit("set HLPD_API_KEY (and HLPD_API_BASE) or pass --api_key/--api_base")

    with open(args.input, encoding="utf-8") as f:
        originals = json.load(f)["original"][: args.n]

    prompts = None
    if args.task in ("adversarial", "polish"):
        with open(args.prompts, encoding="utf-8") as f:
            pool = json.load(f)
        prompts = pool.get("prompts") or pool.get("out_prompt")
    rng = random.Random(args.seed)

    result = {"original": originals, "rewritten": []}
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("original", [])[: len(prev.get("rewritten", []))] == originals[: len(prev.get("rewritten", []))]:
            result["rewritten"] = prev["rewritten"]
            logging.info(f"resuming from {len(result['rewritten'])} revised passages")

    client = make_client(args.api_base, args.api_key)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for idx in range(len(originals)):
        if idx < len(result["rewritten"]) and result["rewritten"][idx]:
            continue  # already done (empty strings mark failures and are retried)
        if args.task == "rewrite":
            system_prompt = REWRITE_PROMPT
        elif args.task == "expand":
            system_prompt = EXPAND_PROMPT.format(style=rng.choice(STYLES))
        else:
            system_prompt = prompts[idx % len(prompts)] if args.prompt_order == "cyclic" else rng.choice(prompts)
        if args.wrap:
            system_prompt += WRAP_INSTRUCTION
        try:
            text = strip_markers(complete(client, args.model, system_prompt, originals[idx], args.temperature))
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

    failed = [i for i, t in enumerate(result["rewritten"]) if not t]
    if failed:  # alignment is kept; empty strings mark failures and are retried on the next run
        logging.warning(f"{len(failed)} passages failed: {failed} -- re-run to retry them")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logging.info(f"saved {len(result['rewritten'])} pairs to {args.output}")


if __name__ == "__main__":
    main()
