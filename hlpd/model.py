"""HLPD scoring model.

``HLPDScorer`` wraps a causal LM (GPT-Neo-2.7B by default) with a LoRA
adapter.  During HLPO training the adapter is optimised so that the model
assigns higher log-probability to human-written text than to its
machine-revised counterpart; a frozen copy of the base model serves as the
reference model ``p_ref``.  At test time the tuned model is used to compute
the HLP-CPC criterion (Sec. 2.3), i.e. the conditional probability curvature
of Fast-DetectGPT evaluated under the human-aligned scorer.
"""
import json
import os
import time

import torch
from torch import nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hlpo_loss import HLPOLoss

MODEL_FULLNAMES = {
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "gpt-j-6B": "EleutherAI/gpt-j-6B",
}
FLOAT16_MODELS = ["gpt-j-6B"]
LEGACY_CKPT_FILES = ("ours-finetuned.pth", "model.bin", "model.pth")


def get_model_fullname(model_name):
    return MODEL_FULLNAMES.get(model_name, model_name)


def from_pretrained(cls, model_name, kwargs, cache_dir):
    """Load from ``cache_dir/<name>`` if it exists locally, else from the Hub."""
    local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    if os.path.isdir(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    if os.path.isdir(model_name):
        return cls.from_pretrained(model_name, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)


def load_tokenizer(model_name, for_dataset="xsum", cache_dir="./models"):
    model_fullname = get_model_fullname(model_name)
    tok_kwargs = {"padding_side": "left" if for_dataset in ["pubmed"] else "right"}
    tokenizer = from_pretrained(AutoTokenizer, model_fullname, tok_kwargs, cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(model_name, device="cuda", cache_dir="./models", torch_dtype=None):
    model_fullname = get_model_fullname(model_name)
    print(f"Loading model {model_fullname} ...")
    model_kwargs = {}
    if model_name in FLOAT16_MODELS:
        model_kwargs["torch_dtype"] = torch.float16
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if "gpt-j" in model_name:
        model_kwargs["revision"] = "float16"
    local_path = os.path.join(cache_dir, model_fullname.split("/")[-1])
    adapter_dir = next((p for p in (model_fullname, local_path) if os.path.exists(os.path.join(p, "adapter_config.json"))), None)
    if adapter_dir is not None:
        # a fully fine-tuned ablation model saved as a PEFT adapter (SFT / ORPO / RLHF)
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, **model_kwargs)
    else:
        model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    start = time.time()
    model.to(device)
    print(f"Moved to {device} ({time.time() - start:.1f}s)")
    return model


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    """Analytic conditional probability curvature (Fast-DetectGPT).

    Returns ``(discrepancy, log_likelihood)`` per sequence.  With the
    human-aligned scorer, *human* text obtains the larger curvature, so the
    HLPD detection score is ``-discrepancy`` (Eq. 12).
    """
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    return discrepancy, log_likelihood.sum(dim=-1)


def hlpd_score_from_logits(logits, labels):
    """HLPD detection score (higher = more likely machine-revised)."""
    crit, _ = get_sampling_discrepancy_analytic(logits, logits, labels)
    return -crit


class HLPDScorer(nn.Module):
    """Scoring model + frozen reference model used for HLPO training and
    HLP-CPC evaluation.

    Parameters
    ----------
    scoring_model_name, reference_model_name : str
        Short names (see ``MODEL_FULLNAMES``) or HF ids / local paths.
    loss : HLPOLoss | None
        Training objective; ``None`` means default HLPO (linear + Dynamic-beta).
    use_lora : bool
        Attach a LoRA adapter (the standard setting).  ``False`` is used to
        evaluate fully fine-tuned ablation models (SFT / RLHF / ORPO).
    share_reference : bool
        Compute ``p_ref`` with the scoring model's LoRA adapter disabled
        instead of loading a second copy of the base model.  The reference
        LoRA is an identity, so ``p_ref`` is unchanged while GPU memory is
        roughly halved (fits 24 GB cards).
    """

    def __init__(self, scoring_model_name="gpt-neo-2.7B", reference_model_name="gpt-neo-2.7B",
                 loss=None, use_lora=True, dataset="xsum", device="cuda", cache_dir="./models",
                 lora_r=8, lora_alpha=32, lora_dropout=0.1, load_reference=True, share_reference=False):
        super().__init__()
        self.device = device
        self.cache_dir = cache_dir
        self.use_lora = use_lora
        self.scoring_model_name = get_model_fullname(scoring_model_name)
        self.reference_model_name = get_model_fullname(reference_model_name)
        self.loss_fn = loss if loss is not None else HLPOLoss()
        self.criterion_fn = get_sampling_discrepancy_analytic

        self.scoring_tokenizer = load_tokenizer(scoring_model_name, dataset, cache_dir)
        self.reference_tokenizer = load_tokenizer(reference_model_name, dataset, cache_dir)

        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                      r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        scoring_model = load_base_model(scoring_model_name, device, cache_dir)
        self.scoring_model = get_peft_model(scoring_model, self.peft_config) if use_lora else scoring_model

        self.share_reference = share_reference and use_lora
        self.reference_model = None
        if load_reference and not self.share_reference:
            reference_model = load_base_model(reference_model_name, device, cache_dir)
            # the reference LoRA is never updated (initialised as identity), so p_ref == base model
            self.reference_model = get_peft_model(reference_model, self.peft_config) if use_lora else reference_model
            for p in self.reference_model.parameters():
                p.requires_grad_(False)

        if use_lora:
            self.scoring_model.print_trainable_parameters()

    # ------------------------------------------------------------------ checkpoints
    def save_pretrained(self, save_directory):
        """Save only the LoRA adapter of the scoring model (+ tokenizer)."""
        os.makedirs(save_directory, exist_ok=True)
        if self.use_lora:
            self.scoring_model.save_pretrained(save_directory)
        else:
            torch.save(self.scoring_model.state_dict(), os.path.join(save_directory, "model.pth"))
        self.scoring_tokenizer.save_pretrained(save_directory)
        with open(os.path.join(save_directory, "hlpd_config.json"), "w") as f:
            json.dump({"scoring_model": self.scoring_model_name, "reference_model": self.reference_model_name,
                       "loss_type": self.loss_fn.loss_type, "dynamic_beta": self.loss_fn.dynamic_beta,
                       "beta": self.loss_fn.beta, "use_lora": self.use_lora}, f, indent=2)

    def from_pretrained(self, load_directory):
        """Load a checkpoint saved by ``save_pretrained`` or a legacy full
        ``state_dict`` checkpoint (``ours-finetuned.pth`` / ``model.bin``)."""
        if not os.path.isdir(load_directory):
            raise ValueError(f"Checkpoint directory {load_directory} does not exist.")
        adapter_file = [f for f in ("adapter_model.safetensors", "adapter_model.bin") if os.path.exists(os.path.join(load_directory, f))]
        if adapter_file and self.use_lora:
            if adapter_file[0].endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(os.path.join(load_directory, adapter_file[0]))
            else:
                state = torch.load(os.path.join(load_directory, adapter_file[0]), map_location=self.device)
            set_peft_model_state_dict(self.scoring_model, state)
            print(f"Loaded LoRA adapter from {load_directory}")
            return
        legacy = [f for f in LEGACY_CKPT_FILES + ("model.pth",) if os.path.exists(os.path.join(load_directory, f))]
        if not legacy:
            raise ValueError(f"No adapter or legacy checkpoint found in {load_directory}")
        state = torch.load(os.path.join(load_directory, legacy[0]), map_location=self.device)
        if self.reference_model is None:  # drop reference weights if we did not load a reference model
            state = {k: v for k, v in state.items() if not k.startswith("reference_model.")}
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"Loaded legacy checkpoint {legacy[0]} ({len(unexpected)} unexpected / {len(missing)} missing keys)")

    # ------------------------------------------------------------------ scoring
    def _logits(self, model, tokenized):
        return model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:, :-1, :]

    def _score(self, tokenized, labels, training):
        """Return ``(crit, seq_logprob, logits)`` from the scoring model (grad on)
        or from the reference model (no grad)."""
        if training:
            logits = self._logits(self.scoring_model, tokenized)
            crit, logprob = self.criterion_fn(logits, logits, labels)
            return crit, logprob, logits
        with torch.no_grad():
            if self.share_reference:
                with self.scoring_model.disable_adapter():
                    logits = self._logits(self.scoring_model, tokenized)
            else:
                ref = self.reference_model if self.reference_model is not None else self.scoring_model
                logits = self._logits(ref, tokenized)
            crit, logprob = self.criterion_fn(logits, logits, labels)
            return crit, logprob, logits

    def forward(self, text):
        """``text = [human_texts, machine_texts]`` (lists of equal length).

        Returns ``crit`` = [ref_human, model_human, ref_machine, model_machine]
        (raw curvature values) and the HLPO ``loss``.
        """
        human_text, machine_text = text[0], text[1]

        tok = self.scoring_tokenizer(human_text, return_tensors="pt", padding=True,
                                     truncation=True, return_token_type_ids=False).to(self.device)
        labels = tok.input_ids[:, 1:]
        ref_h_crit, ref_h_logprob, _ = self._score(tok, labels, training=False)
        h_crit, h_logprob, _ = self._score(tok, labels, training=True)

        tok = self.scoring_tokenizer(machine_text, return_tensors="pt", padding=True,
                                     truncation=True, return_token_type_ids=False).to(self.device)
        labels = tok.input_ids[:, 1:]
        ref_m_crit, ref_m_logprob, _ = self._score(tok, labels, training=False)
        m_crit, m_logprob, _ = self._score(tok, labels, training=True)

        # human text is the *preferred* sample (x_h ≻ x_m)
        loss, pref_rel, dispref_rel, reward_acc, reward_margin = self.loss_fn(
            h_logprob, m_logprob, ref_h_logprob, ref_m_logprob, update_beta=self.training)
        return dict(crit=[ref_h_crit, h_crit, ref_m_crit, m_crit], loss=loss,
                    reward_acc=reward_acc, reward_margin=reward_margin, beta=self.loss_fn.current_beta)

    @torch.no_grad()
    def detect(self, text):
        """HLPD score of a single passage (higher = more likely machine-revised)."""
        tok = self.scoring_tokenizer(text, return_tensors="pt", padding=True,
                                     truncation=True, return_token_type_ids=False).to(self.device)
        labels = tok.input_ids[:, 1:]
        logits = self._logits(self.scoring_model, tok)
        return hlpd_score_from_logits(logits, labels)


def load_inference_model(checkpoint, base_model=None, cache_dir="./models", device="cuda", torch_dtype=None):
    """Load a lightweight inference model (base LM + LoRA adapter) for
    ``evaluate.py`` / ``detect.py`` / the web app.

    ``checkpoint`` is a directory containing ``adapter_model.safetensors``
    (as written by ``HLPDScorer.save_pretrained`` or ``export_inference_ckpt``).
    ``base_model`` overrides the base recorded in ``adapter_config.json``
    (useful when the recorded path is not available on this machine).
    """
    with open(os.path.join(checkpoint, "adapter_config.json")) as f:
        adapter_cfg = json.load(f)
    base_name = base_model or adapter_cfg.get("base_model_name_or_path") or "gpt-neo-2.7B"
    if not os.path.isdir(base_name) and not os.path.isdir(os.path.join(cache_dir, os.path.basename(base_name))):
        # recorded path is unavailable -> fall back to the default base model
        if base_model is None:
            base_name = "gpt-neo-2.7B"
    base = load_base_model(base_name, device="cpu", cache_dir=cache_dir, torch_dtype=torch_dtype)
    model = PeftModel.from_pretrained(base, checkpoint)
    model.to(device).eval()
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    except Exception:  # tokenizer not stored next to the adapter
        tokenizer = load_tokenizer(base_name, "xsum", cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def score_text(model, tokenizer, text, device="cuda"):
    """HLPD score for one passage with an inference model."""
    tok = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                    return_token_type_ids=False).to(device)
    labels = tok.input_ids[:, 1:]
    logits = model(**tok).logits[:, :-1]
    return hlpd_score_from_logits(logits, labels).item()
