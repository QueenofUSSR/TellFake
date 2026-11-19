"""Binary metrics, validation-only threshold fitting, per-source comparisons."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


def choose_threshold(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    candidates = np.flatnonzero(np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1))
    scores = (tpr[candidates] + 1 - fpr[candidates]) / 2
    best = candidates[np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))]
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def choose_fpr_threshold(labels, probabilities, target=0.01):
    """Smallest threshold meeting empirical real FPR under prediction >= t.

    Move above a tied boundary rather than splitting tied scores. A threshold
    just above 1 is valid if saturated real scores require predicting no AI.
    This controls empirical validation FPR, not population/test FPR.
    """
    labels = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=np.float64)
    if not 0 < target < 1 or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid FPR threshold inputs")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probability")
    real = np.sort(scores[labels == 0])[::-1]
    if not len(real):
        raise ValueError("Real validation images required")
    allowed = int(np.floor(target * len(real)))
    return float(np.nextafter(real[allowed], np.inf))


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Metrics require matching, nonempty labels and predictions")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid predicted probability")
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return dict(n=len(labels), n_real=int((labels == 0).sum()), n_ai=int((labels == 1).sum()),
                threshold=float(threshold), accuracy=float(accuracy_score(labels, predictions)),
                balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
                precision=float(precision), recall=float(recall), f1=float(f1),
                auroc=float(roc_auc_score(labels, probabilities)) if both_classes else None,
                average_precision=float(average_precision_score(labels, probabilities)) if both_classes else None,
                real_fpr=float(fp / (fp + tn)) if fp + tn else None,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def grouped_metrics(labels, probabilities, generators, threshold):
    labels, probabilities, generators = map(np.asarray, (labels, probabilities, generators))
    result = {"overall": binary_metrics(labels, probabilities, threshold), "by_generator": {}}
    for generator in sorted(set(generators) - {"Real"}):
        # A fake-only subset cannot measure specificity/AUROC. Pair EACH source
        # with the same real test pool; retain counts and balanced accuracy.
        mask = (generators == "Real") | (generators == generator)
        result["by_generator"][str(generator)] = binary_metrics(labels[mask], probabilities[mask], threshold)
    result["macro_generator_balanced_accuracy"] = float(np.mean([
        m["balanced_accuracy"] for m in result["by_generator"].values()]))
    return result
