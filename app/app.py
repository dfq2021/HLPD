"""HLPD web demo (Gradio).

    python app/app.py --checkpoint models/HLPD-inference          # http://127.0.0.1:7860
    python app/app.py --share                                     # public temporary link
    python app/app.py --mock                                      # UI preview without a model / GPU

The page scores a passage with the HLPO-aligned scorer, converts the HLPD
score into a probability with the reference distributions in
``local_infer_ref`` and shows a per-scenario breakdown.
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402

from hlpd import __version__  # noqa: E402
from hlpd.prob_estimator import SCENARIO_LABELS, SCENARIOS, ProbEstimator  # noqa: E402

TITLE = "HLPD · Human Language Preference Detection"
DESCRIPTION = """
Detects **machine-revised** (polished / rewritten / expanded / adversarially revised) and
**machine-generated** text in the black-box setting. The scoring model is GPT-Neo-2.7B aligned to
*human* language preference with **HLPO**; the score is the human-language-preference conditional
probability curvature (**HLP-CPC**). Longer passages (≥ 50 words) give more reliable results.
"""
LINKS = (
    "[📄 Paper (arXiv 2511.06942)](https://arxiv.org/abs/2511.06942) · "
    "[💻 Code](https://github.com/dfq2021/HLPD) · "
    "[📦 Data](https://github.com/dfq2021/HLPD/tree/main/data)"
)
ABOUT = """
**How it works.** HLPO trains the scorer on pairs *(human text ≻ machine-revised text)* so that it
assigns higher log-probability to human writing. At test time the curvature of the log-probability
landscape around the passage is computed (Fast-DetectGPT style, analytic form); human passages sit at
*positive* curvature under the human-aligned scorer, machine-revised passages at *negative*
curvature. The HLPD score is `−d(x)`; the probability is estimated non-parametrically from the score
distributions of our test sets (`local_infer_ref/`).

**Scenarios.** *polish / rewrite / expand / generate* follow the ImBD benchmark, *adversarial* is our
multi-task adversarial revision benchmark (five-dimensional prompts, GPT-4o / Claude-3.5 /
DeepSeek-R1 / Gemini-2.5-Pro). *all* pools every scenario.

**Limitations.** Very short passages, non-English text (the released scorer is English-only) and
heavily post-edited text may be misjudged. Use the output as evidence, not as proof.
"""


def load_examples():
    """A human passage and a machine-revised one from the released data (if present)."""
    examples = []
    try:
        with open(os.path.join(ROOT, "data/adversarial_multitask/gpt-4o/xsum_adversarial_gpt-4o.raw_data.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        i = random.Random(0).randrange(len(d["original"]))
        examples = [[d["original"][i], "adversarial"], [d["rewritten"][i], "adversarial"]]
    except Exception:
        pass
    if not examples:
        examples = [["Paste a passage of at least 50 words here.", "all"]]
    return examples


class Detector:
    def __init__(self, args):
        self.args = args
        self.mock = args.mock
        self.estimator = ProbEstimator(args.ref_path)
        self.model = self.tokenizer = None
        if not self.mock:
            import torch
            from hlpd.model import load_inference_model
            print("Loading model ...")
            t0 = time.time()
            self.model, self.tokenizer = load_inference_model(
                args.checkpoint, base_model=args.base_model, cache_dir=args.cache_dir, device=args.device,
                torch_dtype=torch.float16 if args.fp16 else None)
            print(f"Model ready ({time.time() - t0:.1f}s)")

    def score(self, text):
        if self.mock:  # UI preview: the built-in examples get the median reference score of their class,
            import numpy as np  # anything else a deterministic pseudo-score
            ex = load_examples()
            if len(ex) > 1 and text == ex[0][0].strip():
                return float(np.median(self.estimator.real["adversarial"]))
            if len(ex) > 1 and text == ex[1][0].strip():
                return float(np.median(self.estimator.fake["adversarial"]))
            return random.Random(text).uniform(-2.5, 2.5)
        from hlpd.model import score_text
        return score_text(self.model, self.tokenizer, text, self.args.device)

    def __call__(self, text, scenario, detail):
        text = (text or "").strip()
        if not text:
            return {"Human-written": 0.5, "Machine-revised": 0.5}, "Please enter a passage.", ""
        n_words = len(text.split())
        t0 = time.time()
        score = self.score(text)
        elapsed = time.time() - t0
        scenario = scenario if scenario in self.estimator.available or scenario == "all" else "all"
        p = self.estimator.prob(score, scenario)
        label = {"Machine-revised / generated": p, "Human-written": 1 - p}
        if p >= 0.8:
            verdict = "🔴 very likely **machine-revised**"
        elif p >= 0.6:
            verdict = "🟠 likely machine-revised"
        elif p > 0.4:
            verdict = "🟡 uncertain"
        elif p > 0.2:
            verdict = "🟢 likely human-written"
        else:
            verdict = "🟢 very likely **human-written**"
        pct_h, pct_m = self.estimator.percentile(score, scenario)
        summary = (
            f"**Verdict:** {verdict}  \n"
            f"**HLPD score (−HLP-CPC):** `{score:+.4f}`  ·  **P(machine | {scenario}):** `{p * 100:.1f}%`  \n"
            f"score is higher than {pct_h * 100:.0f}% of human passages and {pct_m * 100:.0f}% of machine passages "
            f"in the reference set  \n"
            f"{n_words} words · {elapsed * 1000:.0f} ms"
            + ("  \n⚠️ fewer than 50 words – low confidence" if n_words < 50 else "")
            + ("  \n⚠️ **MOCK MODE** – random scores, UI preview only" if self.mock else "")
        )
        table = ""
        if detail:
            rows = ["| scenario | P(machine) | |", "|---|---:|---|"]
            for s, ps in self.estimator.prob_detail(score).items():
                blocks = "█" * int(round(ps * 20)) + "░" * (20 - int(round(ps * 20)))
                rows.append(f"| {s} ({SCENARIO_LABELS[s]}) | {ps * 100:.1f}% | `{blocks}` |")
            table = "\n".join(rows)
        return label, summary, table


def build_ui(detector, preload=True):
    scenarios = ["all"] + [s for s in SCENARIOS if s in detector.estimator.available]
    examples = load_examples()
    initial_text = examples[1][0] if preload and len(examples) > 1 else ""
    initial_scenario = examples[1][1] if preload and len(examples) > 1 and examples[1][1] in scenarios else "all"
    try:
        theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    except Exception:
        theme = None
    with gr.Blocks(title=TITLE, theme=theme) as demo:
        gr.Markdown(f"# {TITLE}" + ("  ·  ⚠️ MOCK MODE" if detector.mock else ""))
        gr.Markdown(LINKS)
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Passage", lines=14, value=initial_text,
                                  placeholder="Paste the text to examine ...")
                with gr.Row():
                    scenario = gr.Dropdown(scenarios, value=initial_scenario, label="Reference scenario")
                    detail = gr.Checkbox(value=True, label="Per-scenario breakdown")
                with gr.Row():
                    run = gr.Button("Detect", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=2):
                label = gr.Label(label="Prediction", num_top_classes=2)
                summary = gr.Markdown()
                table = gr.Markdown()
        gr.Examples(examples=examples, inputs=[text, scenario], label="Examples (human vs. GPT-4o adversarial revision)")
        with gr.Accordion("About HLPD", open=False):
            gr.Markdown(ABOUT)
        gr.Markdown(f"<sub>HLPD v{__version__} · Dai, Jiang, Deng — AAAI 2026 · MIT License</sub>")

        run.click(detector, inputs=[text, scenario, detail], outputs=[label, summary, table])
        text.submit(detector, inputs=[text, scenario, detail], outputs=[label, summary, table])
        clear.click(lambda: ("", None, "", ""), outputs=[text, label, summary, table])
        if preload and initial_text:  # show a worked example as soon as the page opens
            demo.load(detector, inputs=[text, scenario, detail], outputs=[label, summary, table])
    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(description="HLPD web demo")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(ROOT, "models/HLPD-inference"))
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=os.path.join(ROOT, "models"))
    parser.add_argument("--ref_path", type=str, default=os.path.join(ROOT, "local_infer_ref"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--mock", action="store_true", help="UI preview with random scores (no model needed)")
    parser.add_argument("--no_preload", action="store_true", help="do not score the built-in example on page load")
    args = parser.parse_args(argv)

    detector = Detector(args)
    demo = build_ui(detector, preload=not args.no_preload)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, show_api=False)


if __name__ == "__main__":
    main()
