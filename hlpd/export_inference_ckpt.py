"""Export a trained scorer to a lightweight *inference checkpoint*
(LoRA adapter + tokenizer, a few MB) that ``evaluate.py`` / ``detect.py`` /
the web app can load without the reference model.

Works with adapters written by ``hlpd.train`` and with legacy full
``state_dict`` checkpoints (``ours-finetuned.pth`` from the original
experiments, ~20 GB).

    python -m hlpd.export_inference_ckpt --checkpoint ckpt/<run> --output models/HLPD-inference
"""
import argparse
import json
import os

from .model import HLPDScorer, get_model_fullname


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./models/HLPD-inference")
    parser.add_argument("--base_model", type=str, default="gpt-neo-2.7B")
    parser.add_argument("--cache_dir", type=str, default="./models")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--portable_base", action="store_true",
                        help="record the Hub id of the base model instead of a local path")
    args = parser.parse_args(argv)

    scorer = HLPDScorer(args.base_model, args.base_model, device=args.device, cache_dir=args.cache_dir,
                        load_reference=False)
    scorer.from_pretrained(args.checkpoint)
    scorer.save_pretrained(args.output)

    if args.portable_base:
        cfg_path = os.path.join(args.output, "adapter_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["base_model_name_or_path"] = get_model_fullname(args.base_model)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"Inference checkpoint written to {args.output}")


if __name__ == "__main__":
    main()
