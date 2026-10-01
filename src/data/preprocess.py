"""Preprocessing: cleaning -> filtering -> question aggregation -> Task A labels -> grouped split.

Outputs (paths from config.yaml):
    data/processed/rows.parquet        one row per kept (prompt, response, cot) record, with qid + split
    data/processed/questions.parquet   one row per unique normalised question (the evaluation unit)
    data/splits/{train,val,test}_qids.txt
    data/splits/split_manifest.json    seed, ratios, counts, data hash
    results/preprocessing_report.json  every transformation with before/after counts + leakage checks

Two tasks are defined on the same split:
    Task A  provision identification (objective): gold = majority first-cited provision among the
            question's reference answers; only questions where that provision is not already named
            in the question and the references agree (see `task_a_label`).
    Task B  free-form legal QA: all questions; references = all distinct cleaned responses.

Usage:
    python -m src.data.preprocess [--config config.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.data.categories import legal_area
from src.data.provisions import first_provision, label_number_in_text, provision_number, provisions_in_question
from src.data.split import assign_splits, near_duplicate_groups
from src.utils.config import load_config, resolve
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

log = logging.getLogger("preprocess")

# Questions that refer to text the model never receives. "in this context" alone is common in
# self-contained scenarios, so it only counts for short questions; the anaphoric phrases only
# count when the question does not itself name an Act / Article / Section.
ANAPHORA = re.compile(
    r"\b(these provisions|this provision|the said|this article|this section|this act|this clause|"
    r"these requirements|this amendment|the above|mentioned above|the person who was|this chapter|"
    r"this part|this schedule|this rule|under this)\b",
    re.IGNORECASE,
)
SHORT_CONTEXT = re.compile(r"\b(in this context|in the given context|this passage|the passage)\b", re.IGNORECASE)
NAMES_LAW = re.compile(r"\b(Act|Code|Constitution|Article\s+\d+[A-Z]*|Section\s+\d+[A-Z]*|Sanhita)\b", re.IGNORECASE)
TERMINAL = re.compile(r"[.!?)\]\"'*:]\s*$")


def clean_text(text: str) -> str:
    """Unicode NFKC, remove control chars, collapse horizontal whitespace, cap blank lines at one."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_prompt(text: str) -> str:
    """Key for exact-duplicate grouping: case, punctuation and whitespace insensitive."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_orphan(prompt: str) -> bool:
    n_words = len(prompt.split())
    if SHORT_CONTEXT.search(prompt) and n_words < 15:
        return True
    return bool(ANAPHORA.search(prompt)) and not NAMES_LAW.search(prompt)


def task_a_label(question: str, references: list[str], min_agreement: float) -> dict[str, Any]:
    """Derive the Task A gold label from the references' first-cited provisions.

    Every reference casts one vote (None when it cites no resolvable provision). The gold label
    is the most common non-None vote; the question is eligible when that label has at least
    `min_agreement` of all votes, strictly beats every other label, and its provision number is
    not already written in the question, even as a bare number (otherwise it could be copied).
    """
    votes = [first_provision(r, question) for r in references]
    counts = Counter(v for v in votes if v is not None)
    base = {"task_a_label": None, "task_a_agreement": 0.0, "task_a_eligible": False,
            "task_a_all_labels": sorted(counts), "task_a_reason": "no_citation"}
    if not counts:
        return base
    (gold, n_gold), *rest = counts.most_common()
    agreement = n_gold / len(votes)
    base.update(task_a_label=gold, task_a_agreement=round(agreement, 3))
    if rest and rest[0][1] == n_gold:
        base["task_a_reason"] = "tied_labels"
    elif agreement < min_agreement:
        base["task_a_reason"] = "low_agreement"
    elif provision_number(gold) in provisions_in_question(question) or label_number_in_text(gold, question):
        base["task_a_reason"] = "label_in_question"
    else:
        base.update(task_a_eligible=True, task_a_reason="ok")
    return base


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    pcfg, dcfg = cfg["preprocessing"], cfg["data"]
    seed = cfg["seed"]
    set_seed(seed)
    report: dict[str, Any] = {"seed": seed, "config": {"data": dcfg, "preprocessing": pcfg}, "steps": []}

    def step(name: str, before: int, after: int, note: str = "") -> None:
        report["steps"].append({"step": name, "before": before, "after": after, "removed": before - after, "note": note})
        log.info("%-40s %6d -> %6d  %s", name, before, after, note)

    raw_path = resolve(dcfg["raw_file"])
    raw_bytes = raw_path.read_bytes()
    report["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    df = pd.DataFrame(json.loads(raw_bytes.decode("utf-8")))
    df.insert(0, "row_id", np.arange(len(df)))
    n0 = len(df)

    for col in ("prompt", "response", "complex_cot"):
        df[col] = df[col].fillna("").astype(str).map(clean_text)
    step("clean_text (NFKC, control chars, whitespace)", n0, len(df))

    n = len(df)
    df = df[(df["prompt"] != "") & (df["response"] != "")]
    step("drop empty prompt/response", n, len(df))

    n = len(df)
    df = df.drop_duplicates(subset=["prompt", "response", "complex_cot"])
    step("drop exact duplicate rows", n, len(df))

    df["response_words"] = df["response"].str.split().str.len()
    n = len(df)
    df = df[df["response_words"].between(pcfg["min_response_words"], pcfg["max_response_words"])]
    step("response length filter", n, len(df),
         f"keep {pcfg['min_response_words']}..{pcfg['max_response_words']} words")

    df["orphan"] = df["prompt"].map(is_orphan)
    if pcfg["drop_orphan_prompts"]:
        n = len(df)
        df = df[~df["orphan"]]
        step("drop context-dependent (orphan) prompts", n, len(df))

    df["prompt_norm"] = df["prompt"].map(normalize_prompt)
    df["response_truncated"] = ~df["response"].str.contains(TERMINAL)
    df = df.reset_index(drop=True)

    # ---- aggregate to questions ------------------------------------------------------
    grouped = df.groupby("prompt_norm", sort=False)
    q = grouped.agg(
        prompt=("prompt", "first"),
        n_rows=("row_id", "size"),
        row_ids=("row_id", list),
    ).reset_index()
    # Distinct references per question (normalised to avoid counting whitespace variants twice).
    q["references"] = grouped["response"].agg(
        lambda s: list(dict.fromkeys(s.tolist()))
    ).values
    q["n_refs"] = q["references"].str.len()
    q.insert(0, "qid", np.arange(len(q)))
    df["qid"] = df["prompt_norm"].map(dict(zip(q["prompt_norm"], q["qid"])))
    report["n_questions"] = len(q)
    report["questions_with_multiple_references"] = int((q["n_refs"] > 1).sum())
    log.info("unique questions: %d (%d with >1 distinct reference)", len(q), report["questions_with_multiple_references"])

    # ---- Task A labels -------------------------------------------------------------------
    labels = [task_a_label(p, refs, pcfg["task_a_min_agreement"]) for p, refs in zip(q["prompt"], q["references"])]
    q = pd.concat([q, pd.DataFrame(labels)], axis=1)
    q["task_a_tier"] = np.where(
        ~q["task_a_eligible"], "none",
        np.where(q["task_a_label"].fillna("").str.startswith("art:"), "constitution", "statute"),
    )
    report["task_a"] = {
        "eligible": int(q["task_a_eligible"].sum()),
        "reason_counts": q["task_a_reason"].value_counts().to_dict(),
        "tier_counts": q.loc[q["task_a_eligible"], "task_a_tier"].value_counts().to_dict(),
        "distinct_labels": int(q.loc[q["task_a_eligible"], "task_a_label"].nunique()),
    }
    log.info("Task A: %s", report["task_a"])

    # ---- other question-level fields -----------------------------------------------------
    q["legal_area"] = q["prompt"].map(legal_area)
    q["prompt_words"] = q["prompt"].str.split().str.len()

    # ---- grouped split ---------------------------------------------------------------------
    q["group_id"] = near_duplicate_groups(q["prompt"].tolist(), dcfg["near_dup_threshold"])
    gsize = q.groupby("group_id").size()
    report["near_dup_components"] = {
        "threshold": dcfg["near_dup_threshold"],
        "n_components": int(len(gsize)),
        "largest": int(gsize.max()),
        "size_quantiles": {f"p{p}": float(np.percentile(gsize, p)) for p in (50, 90, 99)},
        "questions_in_multi_question_components": int(gsize[gsize > 1].sum()),
    }
    # A component's stratum is the Task A tier of its "best" member (constitution > statute > none).
    tier_rank = {"constitution": 2, "statute": 1, "none": 0}
    comp = q.assign(r=q["task_a_tier"].map(tier_rank)).groupby("group_id").agg(
        n_questions=("qid", "size"), r=("r", "max")).reset_index()
    comp["stratum"] = comp["r"]
    mapping = assign_splits(comp, dcfg["split_ratios"], seed)
    q["split"] = q["group_id"].map(mapping)
    df["split"] = df["qid"].map(dict(zip(q["qid"], q["split"])))

    report["split_counts"] = {
        "questions": q["split"].value_counts().to_dict(),
        "rows": df["split"].value_counts().to_dict(),
        "task_a_questions": q[q["task_a_eligible"]]["split"].value_counts().to_dict(),
        "task_a_by_tier": q[q["task_a_eligible"]].groupby(["split", "task_a_tier"]).size()
        .unstack(fill_value=0).to_dict(orient="index"),
    }
    log.info("split counts: %s", report["split_counts"])

    report["leakage_checks"] = leakage_checks(q, df, dcfg["near_dup_threshold"])
    log.info("leakage: %s", report["leakage_checks"])

    # ---- save -------------------------------------------------------------------------------
    proc, splits = resolve(dcfg["processed_dir"]), resolve(dcfg["splits_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    df[["row_id", "qid", "split", "prompt", "response", "complex_cot", "response_words",
        "response_truncated", "orphan"]].to_parquet(proc / "rows.parquet", index=False)
    q.drop(columns=["prompt_norm"]).to_parquet(proc / "questions.parquet", index=False)
    for s in ("train", "val", "test"):
        (splits / f"{s}_qids.txt").write_text("\n".join(map(str, q.loc[q["split"] == s, "qid"])), encoding="utf-8")
    manifest = {
        "seed": seed, "ratios": dcfg["split_ratios"], "near_dup_threshold": dcfg["near_dup_threshold"],
        "raw_sha256": report["raw_sha256"], "counts": report["split_counts"],
        "questions_sha256": hashlib.sha256(pd.util.hash_pandas_object(q[["qid", "split"]]).values.tobytes()).hexdigest(),
    }
    (splits / "split_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    out = resolve(cfg["paths"]["results"]) / "preprocessing_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("saved processed data, splits and %s", out)
    return report


def leakage_checks(q: pd.DataFrame, df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Verify the split: no shared questions, no near-duplicate prompts, shared-response count."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    res: dict[str, Any] = {}
    by = {s: set(q.loc[q["split"] == s, "prompt_norm"]) for s in ("train", "val", "test")}
    res["exact_question_overlap"] = {
        "train_test": len(by["train"] & by["test"]), "train_val": len(by["train"] & by["val"]),
        "val_test": len(by["val"] & by["test"]),
    }
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(q["prompt"])
    Xtr = vec.transform(q.loc[q["split"] == "train", "prompt"])
    for s in ("val", "test"):
        Xs = vec.transform(q.loc[q["split"] == s, "prompt"])
        nn = np.concatenate([(Xs[i:i + 2000] @ Xtr.T).max(axis=1).toarray().ravel()
                             for i in range(0, Xs.shape[0], 2000)])
        res[f"{s}_to_train_max_cosine"] = {
            "max": round(float(nn.max()), 4),
            "p99": round(float(np.percentile(nn, 99)), 4),
            "p50": round(float(np.percentile(nn, 50)), 4),
            f"n_at_or_above_{threshold}": int((nn >= threshold).sum()),
        }
    # Identical answers across splits indicate templated responses (reported, not removed).
    resp = {s: set(df.loc[df["split"] == s, "response"]) for s in ("train", "test")}
    res["identical_response_train_test"] = len(resp["train"] & resp["test"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    setup_logging(resolve("experiments/logs/preprocess.log"))
    run(load_config(resolve(args.config)))


if __name__ == "__main__":
    main()
