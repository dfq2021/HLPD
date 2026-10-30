"""Training (HLPO) and evaluation (HLP-CPC) loops."""
import datetime
import json
import os
import time

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .metrics import get_precision_recall_metrics, get_roc_metrics


def _writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception:  # tensorboard not installed
        return None


def dataset_tag(path):
    """``data/.../xsum_rewrite_gpt-4o.raw_data.json`` -> ``xsum_rewrite_gpt-4o``."""
    name = os.path.basename(path)
    for suffix in (".raw_data.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@torch.no_grad()
def evaluate(model, dataset, device="cuda", desc="Evaluating"):
    """Score every pair with the HLP-CPC criterion and compute AUROC / PR-AUC.

    The stored scores are ``-d(x)`` (Eq. 12): larger = more likely machine-revised.
    """
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    crit_real, crit_sampled, losses = [], [], []
    start = time.time()
    for batch in tqdm.tqdm(loader, desc=desc):
        out = model(batch)
        losses.append(float(out["loss"]))
        crit_real.extend([-c for c in out["crit"][1].tolist()])
        crit_sampled.extend([-c for c in out["crit"][3].tolist()])
    elapsed = time.time() - start

    fpr, tpr, roc_auc = get_roc_metrics(crit_real, crit_sampled)
    p, r, pr_auc = get_precision_recall_metrics(crit_real, crit_sampled)
    print(f"Total time: {elapsed:.1f}s ({elapsed / max(1, len(dataset)):.3f}s / pair)")
    print(f"val_loss: {np.mean(losses):.6f}")
    print(f"val_ROC_AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    print(f"val_Real_mean/std: {np.mean(crit_real):.2f}/{np.std(crit_real):.2f}, "
          f"val_Samples_mean/std: {np.mean(crit_sampled):.2f}/{np.std(crit_sampled):.2f}")
    print("=" * 10)
    return {
        "name": "HLPD",
        "val_loss": float(np.mean(losses)),
        "val_ROC_AUC": roc_auc,
        "val_PR_AUC": pr_auc,
        "val_crit_real_mean": float(np.mean(crit_real)),
        "val_crit_real_std": float(np.std(crit_real)),
        "val_crit_sampled_mean": float(np.mean(crit_sampled)),
        "val_crit_sampled_std": float(np.std(crit_sampled)),
        "n_pairs": len(dataset),
        "time_seconds": elapsed,
        "crit_real": crit_real,
        "crit_sampled": crit_sampled,
        "fpr": fpr,
        "tpr": tpr,
        "precision": p,
        "recall": r,
    }


def save_result(result, output_dir, eval_path, extra=None, suffix=""):
    os.makedirs(output_dir, exist_ok=True)
    result = dict(result)
    result["eval_dataset"] = eval_path
    if extra:
        result.update(extra)
    out = os.path.join(output_dir, f"{dataset_tag(eval_path)}{suffix}.hlpd.json")
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"Results written to {out}")
    return out


def train(model, train_dataset, val_dataset, args, device="cuda", ckpt_dir="./ckpt"):
    """HLPO fine-tuning.  Returns ``(model, last_val_result)``."""
    run_name = f"{args.task_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = _writer(os.path.join(args.log_dir, run_name, "train"))

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    accumulation_steps = max(1, args.a)
    total_updates = max(1, (len(loader) * args.epochs) // accumulation_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=0)
    use_amp = (not args.no_amp) and device.startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.to(device)
    step, update = 0, 0
    last_val = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        crit_real, crit_sampled, epoch_losses, betas = [], [], [], []
        running = torch.tensor(0.0, device=device)
        start = time.time()
        for batch in tqdm.tqdm(loader, desc=f"HLPO epoch {epoch}"):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(batch)
            crit_real.extend([-c for c in out["crit"][1].tolist()])
            crit_sampled.extend([-c for c in out["crit"][3].tolist()])
            betas.append(float(out["beta"]))
            running = running + out["loss"].to(torch.float32) / accumulation_steps
            step += 1
            if step % accumulation_steps == 0:
                scaler.scale(running).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                update += 1
                epoch_losses.append(running.item())
                if writer:
                    writer.add_scalar("Loss/train", running.item(), update)
                    writer.add_scalar("beta", betas[-1], update)
                    writer.add_scalar("reward_margin", float(out["reward_margin"]), update)
                    writer.add_scalar("lr", scheduler.get_last_lr()[0], update)
                running = torch.tensor(0.0, device=device)

        fpr, tpr, roc_auc = get_roc_metrics(crit_real, crit_sampled)
        _, _, pr_auc = get_precision_recall_metrics(crit_real, crit_sampled)
        print(f"Epoch {epoch}: {time.time() - start:.0f}s  loss={np.mean(epoch_losses):.4f}  "
              f"train ROC AUC={roc_auc:.4f}  PR AUC={pr_auc:.4f}  beta={betas[-1]:.3f}")
        print(f"Real mean/std: {np.mean(crit_real):.2f}/{np.std(crit_real):.2f}, "
              f"Samples mean/std: {np.mean(crit_sampled):.2f}/{np.std(crit_sampled):.2f}")
        if writer:
            writer.add_scalar("Loss/epoch", float(np.mean(epoch_losses)), epoch)
            writer.add_scalar("ROC_AUC/epoch", roc_auc, epoch)
            writer.add_scalar("PR_AUC/epoch", pr_auc, epoch)

        if val_dataset is not None and ((epoch + 1) % args.val_freq == 0 or epoch + 1 == args.epochs):
            print(f"Evaluating on {dataset_tag(args.eval_dataset)} ...")
            last_val = evaluate(model, val_dataset, device)
            if writer:
                writer.add_scalar("val_ROC_AUC/epoch", last_val["val_ROC_AUC"], epoch)
                writer.add_scalar("val_PR_AUC/epoch", last_val["val_PR_AUC"], epoch)
            if args.save_every_eval:
                model.save_pretrained(os.path.join(ckpt_dir, f"epoch{epoch + 1}"))

    model.save_pretrained(ckpt_dir)
    print(f"Saved HLPO adapter to {ckpt_dir}")
    if writer:
        writer.close()
    return model, last_val


def run(model, train_dataset, val_dataset, args, device="cuda", ckpt_dir="./ckpt"):
    """Entry used by ``train.py``: optional pre-tuning evaluation, HLPO
    training and final evaluation.  Returns the final evaluation dict."""
    result = None
    if args.ebt or args.eval_only:
        print("Evaluating model before tuning ...")
        result = evaluate(model, val_dataset, device)
        save_result(result, args.output_dir, args.eval_dataset,
                    extra={"checkpoint": args.from_pretrained or "base"},
                    suffix="" if args.eval_only else ".before_tuning")
        if args.eval_only:
            return result

    print("Fine-tuning scoring model with HLPO ...")
    model, result = train(model, train_dataset, val_dataset, args, device=device, ckpt_dir=ckpt_dir)
    if result is not None:
        save_result(result, args.output_dir, args.eval_dataset, extra={"checkpoint": ckpt_dir})
    return result
