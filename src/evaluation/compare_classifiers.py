"""Paired significance test: Transformer (primary seed) vs TF-IDF+LR on Task A test.

Usage:
    python -m src.evaluation.compare_classifiers   # -> results/significance_transformer_vs_lr.json
"""
from __future__ import annotations

import json

import pandas as pd

from src.evaluation.metrics import mcnemar_exact
from src.utils.config import load_config, resolve


def main() -> None:
    res = resolve(load_config()["paths"]["results"])
    lr = pd.read_parquet(res / "predictions" / "baseline_tfidf_lr_test.parquet")
    bt = pd.read_parquet(res / "predictions" / "transformer_test.parquet")
    m = lr[["qid", "label", "pred", "tier"]].merge(bt[["qid", "pred"]], on="qid", suffixes=("_lr", "_bert"))
    out = {}
    for name, d in [("all", m), *((t, m[m["tier"] == t]) for t in ("constitution", "statute"))]:
        out[name] = {"system_a": "tfidf_lr", "system_b": "inlegalbert_seed42",
                     **mcnemar_exact(d["pred_lr"] == d["label"], d["pred_bert"] == d["label"])}
    (res / "significance_transformer_vs_lr.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
