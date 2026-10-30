"""Threshold-free detection metrics (AUROC / PR-AUC).

Convention: ``real_preds`` are scores of human-written passages (label 0) and
``sample_preds`` scores of machine-revised passages (label 1); a higher score
means "more likely machine-revised".
"""
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), list(real_preds) + list(sample_preds))
    return fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr))


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  list(real_preds) + list(sample_preds))
    return precision.tolist(), recall.tolist(), float(auc(recall, precision))


def tpr_at_fpr(real_preds, sample_preds, target_fpr=0.05):
    """True-positive rate at a fixed false-positive rate."""
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), list(real_preds) + list(sample_preds))
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    return float(np.interp(target_fpr, fpr, tpr))


def mean_confidence_interval(values, z=1.96):
    """Mean and half-width of the normal-approximation 95% CI over runs."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), 0.0
    stderr = values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(z * stderr)
