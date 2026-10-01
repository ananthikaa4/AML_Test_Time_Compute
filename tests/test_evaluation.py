"""Tests for evaluation metrics and result aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import bootstrap_ci, classification_metrics, grouped_metrics


def test_accuracy_and_ceiling():
    m = classification_metrics(["a", "b", "c", "d"], ["a", "b", "x", None], train_labels=["a", "b", "c"])
    assert m["accuracy"] == 0.5
    assert m["ceiling"] == 0.75
    assert m["no_prediction_rate"] == 0.25


def test_macro_f1_only_averages_gold_labels():
    # gold labels a, b; predicted-only label z lowers precision of nothing gold -> macro over {a, b}
    m = classification_metrics(["a", "a", "b"], ["a", "z", "b"])
    # a: P=1, R=0.5 -> F1=2/3 ; b: F1=1  -> macro = 5/6
    assert m["macro_f1"] == pytest.approx(5 / 6)


def test_topk():
    m = classification_metrics(["a", "b"], ["x", "b"], topk=[["x", "y", "a"], ["b"]], ks=(1, 3))
    assert m["top1"] == 0.5 and m["top3"] == 1.0


def test_bootstrap_ci_contains_mean_and_is_ordered():
    correct = np.array([1.0] * 70 + [0.0] * 30)
    lo, hi = bootstrap_ci(correct)
    assert lo < 0.7 < hi and 0.55 < lo and hi < 0.85


def test_bootstrap_ci_deterministic():
    c = np.random.default_rng(1).integers(0, 2, 50).astype(float)
    assert bootstrap_ci(c, seed=3) == bootstrap_ci(c, seed=3)


def test_grouped_metrics_aggregation():
    df = pd.DataFrame({"label": ["a", "b", "c", "d"], "pred": ["a", "x", "c", "d"], "tier": ["t1", "t1", "t2", "t2"]})
    g = grouped_metrics(df, ["tier"])
    acc = dict(zip(g["value"], g["accuracy"]))
    assert acc == {"all": 0.75, "t1": 0.5, "t2": 1.0}
    assert g.loc[g["value"] == "all", "n"].item() == 4
