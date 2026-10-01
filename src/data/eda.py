"""Exploratory data analysis on the processed data.

Reads data/processed/{rows,questions}.parquet (run src.data.preprocess first) and writes
plots/eda/*.png, results/eda_summary.json and results/audit/task_a_label_audit_sample.csv.

Usage:
    python -m src.data.eda
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.utils.config import load_config, resolve
from src.utils.logging import setup_logging
from src.utils.plotting import SERIES, TEXT_SECONDARY, apply_style, save

import matplotlib.pyplot as plt  # noqa: E402  (after plotting sets the backend)

log = logging.getLogger("eda")
SPLITS = ["train", "val", "test"]
# Words that are ubiquitous in legal questions and carry no topical information.
QUESTION_STOP = ENGLISH_STOP_WORDS | {"explain", "discuss", "does", "legal", "law", "india", "indian", "act",
                                      "provisions", "provision", "section", "article", "case", "role", "concept",
                                      "significance", "describe", "analyze", "provide", "example", "state"}


def length_plots(rows: pd.DataFrame, q: pd.DataFrame, out: Any) -> dict[str, Any]:
    fields = [("Question", q["prompt"].str.split().str.len()),
              ("Reference answer", rows["response"].str.split().str.len()),
              ("Reasoning trace (complex_cot)", rows["complex_cot"].str.split().str.len())]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    stats = {}
    for ax, (name, s) in zip(axes, fields):
        cap = float(np.percentile(s, 99))
        ax.hist(s.clip(upper=cap), bins=50, color=SERIES[0], edgecolor="white", linewidth=0.5)
        ax.axvline(s.median(), color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel(f"words (clipped at p99 = {cap:.0f})")
        ax.text(0.97, 0.92, f"median {s.median():.0f}", transform=ax.transAxes, ha="right", color=TEXT_SECONDARY, fontsize=9)
        stats[name] = {"median": float(s.median()), "mean": round(float(s.mean()), 1), "p99": cap}
    axes[0].set_ylabel("count")
    save(fig, out / "length_distributions.png")
    return stats


def references_plot(q: pd.DataFrame, out: Any) -> dict[str, int]:
    counts = q["n_refs"].clip(upper=6).value_counts().sort_index()
    labels = [str(i) if i < 6 else "6+" for i in counts.index]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(labels, counts.values, color=SERIES[0], width=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("distinct reference answers per question")
    ax.set_ylabel("questions (log scale)")
    ax.set_title("Duplicate questions with differing answers")
    for x, v in zip(labels, counts.values):
        ax.text(x, v * 1.1, f"{v:,}", ha="center", fontsize=8, color=TEXT_SECONDARY)
    save(fig, out / "references_per_question.png")
    return {k: int(v) for k, v in zip(labels, counts.values)}


def area_plot(q: pd.DataFrame, out: Any) -> dict[str, dict[str, int]]:
    all_c = q["legal_area"].value_counts()
    a_c = q.loc[q["task_a_eligible"], "legal_area"].value_counts().reindex(all_c.index, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, s, title in [(axes[0], all_c, "All questions (Task B)"), (axes[1], a_c, "Task A questions")]:
        ax.barh(s.index[::-1], s.values[::-1], color=SERIES[0], height=0.7)
        ax.set_title(title)
        ax.set_xlabel("questions")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Heuristic legal area (keyword rules on question text)", color=TEXT_SECONDARY, fontsize=10)
    save(fig, out / "legal_area_distribution.png")
    return {"all": all_c.to_dict(), "task_a": a_c.to_dict()}


def task_a_plots(q: pd.DataFrame, out: Any) -> dict[str, Any]:
    reasons = q["task_a_reason"].value_counts()
    elig = q[q["task_a_eligible"]]
    freq = elig["task_a_label"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].barh(reasons.index[::-1], reasons.values[::-1], color=SERIES[0], height=0.6)
    axes[0].set_title("Task A eligibility of all questions")
    axes[0].set_xlabel("questions")
    axes[0].grid(axis="y", visible=False)
    for tier, color in [("constitution", SERIES[0]), ("statute", SERIES[1])]:
        f = elig.loc[elig["task_a_tier"] == tier, "task_a_label"].value_counts().values
        axes[1].loglog(np.arange(1, len(f) + 1), f, color=color, label=f"{tier} ({len(f)} labels)")
    axes[1].set_title("Task A label frequency (long tail)")
    axes[1].set_xlabel("label rank")
    axes[1].set_ylabel("questions")
    axes[1].legend()
    save(fig, out / "task_a_labels.png")

    top = freq.head(20)
    return {
        "eligibility_reasons": reasons.to_dict(),
        "n_labels": int(len(freq)),
        "labels_with_1_question": int((freq == 1).sum()),
        "labels_with_ge_5_questions": int((freq >= 5).sum()),
        "share_of_questions_covered_by_top_100_labels": round(float(freq.head(100).sum() / freq.sum()), 3),
        "top_20_labels": top.to_dict(),
        "test_labels_unseen_in_train": int(
            (~elig.loc[elig["split"] == "test", "task_a_label"].isin(set(elig.loc[elig["split"] == "train", "task_a_label"]))).sum()
        ),
        "agreement_when_multi_reference": elig.loc[elig["n_refs"] > 1, "task_a_agreement"].describe().round(3).to_dict(),
    }


def top_terms_plot(q: pd.DataFrame, out: Any, k: int = 25) -> dict[str, int]:
    tokens = Counter(
        t for p in q["prompt"].str.lower() for t in re.findall(r"[a-z]{3,}", p) if t not in QUESTION_STOP
    )
    top = tokens.most_common(k)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color=SERIES[0], height=0.7)
    ax.set_title(f"Top {k} question terms (stop words removed)")
    ax.set_xlabel("occurrences")
    ax.grid(axis="y", visible=False)
    save(fig, out / "top_question_terms.png")
    return dict(top)


def split_plot(q: pd.DataFrame, out: Any) -> dict[str, Any]:
    tiers = ["constitution", "statute", "none"]
    tab = q.groupby(["split", "task_a_tier"]).size().unstack(fill_value=0).reindex(index=SPLITS, columns=tiers, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(SPLITS))
    w = 0.27
    names = {"constitution": "Task A: constitution", "statute": "Task A: statute", "none": "Task B only"}
    for i, t in enumerate(tiers):
        vals = tab[t].values
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=SERIES[i], label=names[t])
    ax.set_xticks(x, SPLITS)
    ax.set_yscale("log")
    ax.set_ylabel("questions (log scale)")
    ax.set_title("Grouped split composition")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, out / "split_composition.png")
    return tab.to_dict(orient="index")


def leakage_plot(q: pd.DataFrame, threshold: float, out: Any) -> None:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(q["prompt"])
    Xtr = vec.transform(q.loc[q["split"] == "train", "prompt"])
    Xte = vec.transform(q.loc[q["split"] == "test", "prompt"])
    nn = np.concatenate([(Xte[i:i + 2000] @ Xtr.T).max(axis=1).toarray().ravel() for i in range(0, Xte.shape[0], 2000)])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(nn, bins=60, color=SERIES[0], edgecolor="white", linewidth=0.5)
    ax.axvline(threshold, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.text(threshold, ax.get_ylim()[1] * 0.92, f" grouping threshold {threshold}", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlabel("cosine similarity of test question to its nearest train question")
    ax.set_ylabel("test questions")
    ax.set_title("Split leakage check")
    save(fig, out / "test_train_similarity.png")


def label_audit_sample(q: pd.DataFrame, path: Any, n_per_tier: int = 30, seed: int = 0) -> None:
    """Sample TRAIN Task A items for manual label verification (never test, to avoid peeking)."""
    tr = q[(q["split"] == "train") & q["task_a_eligible"]]
    parts = [tr[tr["task_a_tier"] == t].sample(n_per_tier, random_state=seed) for t in ("constitution", "statute")]
    s = pd.concat(parts)
    s = s.assign(reference_excerpt=s["references"].map(lambda r: r[0][:600]))
    s = s[["qid", "task_a_tier", "task_a_label", "prompt", "reference_excerpt"]]
    s["label_correct_manual"] = ""  # to be filled in by a human: yes / no / unsure
    s["notes"] = ""
    s.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    setup_logging(resolve("experiments/logs/eda.log"))
    cfg = load_config()
    apply_style()
    proc = resolve(cfg["data"]["processed_dir"])
    rows = pd.read_parquet(proc / "rows.parquet")
    q = pd.read_parquet(proc / "questions.parquet")
    out = resolve(cfg["paths"]["plots"]) / "eda"

    summary: dict[str, Any] = {
        "n_rows": len(rows), "n_questions": len(q),
        "lengths_words": length_plots(rows, q, out),
        "references_per_question": references_plot(q, out),
        "legal_area": area_plot(q, out),
        "task_a": task_a_plots(q, out),
        "top_question_terms": top_terms_plot(q, out),
        "split_by_tier": split_plot(q, out),
        "response_truncated_rows": int(rows["response_truncated"].sum()),
    }
    leakage_plot(q, cfg["data"]["near_dup_threshold"], out)
    audit_dir = resolve(cfg["paths"]["results"]) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    label_audit_sample(q, audit_dir / "task_a_label_audit_sample.csv", seed=cfg["seed"])
    (resolve(cfg["paths"]["results"]) / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("EDA written to %s", out)


if __name__ == "__main__":
    main()
