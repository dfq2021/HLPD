"""Mean ROC curve with a 95% confidence band over several seeds
(paper Appendix A, Fig. 4) and the corresponding AUROC bar chart (Fig. 5).

    python tools/plot_roc_ci.py --runs results/paper/multi_seed_adversarial_gemini-2.5-pro \
        --output figures/roc_ci.pdf
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc  # noqa: E402

plt.rcParams["font.family"] = "serif"


def load_curves(folder):
    curves = []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        curves.append((np.asarray(d["fpr"]), np.asarray(d["tpr"]), float(d.get("val_ROC_AUC", 0))))
    return curves


def mean_roc(curves, n_grid=200):
    grid = np.linspace(0, 1, n_grid)
    tprs = []
    for fpr, tpr, _ in curves:
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    tprs = np.asarray(tprs)
    mean = tprs.mean(axis=0)
    stderr = tprs.std(axis=0, ddof=1) / np.sqrt(len(tprs)) if len(tprs) > 1 else np.zeros_like(mean)
    upper = np.minimum(mean + 1.96 * stderr, 1)
    lower = np.maximum(mean - 1.96 * stderr, 0)
    return grid, mean, lower, upper


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, required=True,
                        help="directory with one sub-directory per dataset, each holding per-seed *.json results")
    parser.add_argument("--output", type=str, default="figures/roc_ci.pdf")
    parser.add_argument("--title", type=str, default="ROC with 95% CI")
    args = parser.parse_args(argv)

    datasets = sorted(d for d in os.listdir(args.runs) if os.path.isdir(os.path.join(args.runs, d)))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4), squeeze=False)
    bars = []
    for ax, ds in zip(axes[0], datasets):
        curves = load_curves(os.path.join(args.runs, ds))
        if not curves:
            continue
        grid, mean, lower, upper = mean_roc(curves)
        aucs = [c[2] for c in curves]
        m, se = np.mean(aucs), (np.std(aucs, ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else 0)
        bars.append((ds, m, 1.96 * se))
        ax.plot(grid, mean, color="#0072B2", lw=2, label=f"mean ROC (AUC={auc(grid, mean):.3f})")
        ax.fill_between(grid, lower, upper, color="#0072B2", alpha=0.2,
                        label=f"95% CI [{auc(grid, lower):.3f}, {auc(grid, upper):.3f}]")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_title(f"{ds} (n={len(curves)} seeds)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"saved {args.output}")

    # AUROC bars with CI
    fig2, ax2 = plt.subplots(figsize=(1.6 * len(bars) + 2, 4))
    names = [b[0] for b in bars]
    means = [b[1] for b in bars]
    cis = [b[2] for b in bars]
    ax2.bar(names, means, yerr=cis, capsize=6, color="steelblue", edgecolor="black", width=0.5)
    for i, (m, ci) in enumerate(zip(means, cis)):
        ax2.text(i, m + ci + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylim(max(0.0, min(means) - 0.1), 1.0)
    ax2.set_ylabel("AUROC")
    ax2.grid(axis="y", ls="--", alpha=0.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    out2 = os.path.splitext(args.output)[0] + "_bars" + os.path.splitext(args.output)[1]
    fig2.tight_layout()
    fig2.savefig(out2, dpi=300)
    print(f"saved {out2}")
    for ds, m, ci in bars:
        print(f"{ds}: AUROC {m:.4f} +/- {ci:.4f}")


if __name__ == "__main__":
    main()
