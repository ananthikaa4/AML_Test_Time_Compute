"""Evaluation metrics for Task A (single-label provision identification).

Conventions (stated once, used everywhere):
* accuracy      exact match of predicted vs gold provision key (== exact match / micro-F1)
* macro_f1      unweighted mean F1 over the gold labels present in the evaluated set
                (predicted-only labels are not averaged in; they still lower precision)
* weighted_f1   F1 averaged over the same labels weighted by gold support
* top{k}        gold label among the k highest-scored predictions (classifiers only)
* ceiling       share of gold labels that exist in the training label set (upper bound
                for any closed-set classifier)
* 95% CI        percentile bootstrap over questions (default 2000 resamples)
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if len(correct) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    means = correct[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    topk: Sequence[Sequence[str]] | None = None,
    train_labels: Iterable[str] | None = None,
    ks: tuple[int, ...] = (5,),
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray([p if p is not None else "<none>" for p in y_pred], dtype=object)
    correct = (y_true == y_pred).astype(float)
    labels = sorted(set(y_true))
    lo, hi = bootstrap_ci(correct)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "acc_ci_low": lo,
        "acc_ci_high": hi,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "no_prediction_rate": float((y_pred == "<none>").mean()),
    }
    if topk is not None:
        for k in ks:
            out[f"top{k}"] = float(np.mean([t in list(c)[:k] for t, c in zip(y_true, topk)]))
    if train_labels is not None:
        seen = set(train_labels)
        out["ceiling"] = float(np.mean([t in seen for t in y_true]))
    return out


def grouped_metrics(df: pd.DataFrame, group_cols: Sequence[str], **kw) -> pd.DataFrame:
    """Metrics overall and per value of each column in `group_cols`.

    `df` needs columns: label, pred, (optional) topk. Returns a long frame with columns
    group, value, and the metric fields.
    """
    rows = []

    def _one(sub: pd.DataFrame, group: str, value: str) -> None:
        m = classification_metrics(sub["label"].tolist(), sub["pred"].tolist(),
                                   topk=sub["topk"].tolist() if "topk" in sub else None, **kw)
        rows.append({"group": group, "value": value, **m})

    _one(df, "all", "all")
    for col in group_cols:
        for val, sub in df.groupby(col):
            _one(sub, col, str(val))
    return pd.DataFrame(rows)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """Exact (binomial) McNemar test for two systems scored on the same questions."""
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"n": int(len(a)), "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}
