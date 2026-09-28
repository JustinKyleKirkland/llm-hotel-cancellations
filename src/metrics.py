"""Metrics for binary cancel probabilities: accuracy, F1, ROC-AUC, Brier, ECE, coverage vs accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score


def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error on the top-label confidence (max(p, 1-p)), equal-width bins --
    the usual definition (Guo et al., 2017). 0 = the confidence means what it says."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(y, p, n_bins: int = 10):
    """Predicted P(cancel) vs observed cancel rate in equal-count bins (deciles of the prediction),
    so a method whose probabilities all sit near 0 still gets ten points instead of one."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            rows.append({"bin_lo": float(p[chunk].min()), "bin_hi": float(p[chunk].max()), "n": int(len(chunk)),
                         "mean_pred": float(p[chunk].mean()), "observed": float(y[chunk].mean())})
    return rows


def coverage_curve(y, p, thresholds):
    """Decide automatically when confidence max(p, 1-p) >= t; the rest goes to a human."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    rows = []
    for t in thresholds:
        m = conf >= t - 1e-12
        rows.append({"threshold": float(t), "coverage": float(m.mean()),
                     "n_auto": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean()) if m.any() else None})
    return rows


def coverage_at_risk(y, p, max_error: float) -> float:
    """Largest share of bookings you can decide automatically, taking the most confident first,
    while keeping the error rate on the decided share <= max_error."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    wrong = ((p >= 0.5).astype(int) != y)[order]
    err = np.cumsum(wrong) / np.arange(1, len(y) + 1)
    ok = np.nonzero(err <= max_error)[0]
    return float((ok.max() + 1) / len(y)) if len(ok) else 0.0


def compute(y, p, cfg) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p, cfg["experiment"]["ece_bins"]),
        "pred_cancel_rate": float(pred.mean()),
        "coverage_at_5pct_error": coverage_at_risk(y, p, 0.05),
        "coverage_at_10pct_error": coverage_at_risk(y, p, 0.10),
        "coverage": coverage_curve(y, p, cfg["experiment"]["thresholds"]),
        "reliability": reliability(y, p),
    }
    return out
