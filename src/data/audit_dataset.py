"""Dataset audit for the Kaggle `alpie/indian-law` dataset.

Produces a machine-readable JSON summary and a human-readable Markdown report.
Everything reported here is computed from the raw file; nothing is assumed.

Usage:
    python -m src.data.audit_dataset --input data/raw/Alpie-core_core_indian_law.json \
        --out-dir results/audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("audit")
# pandas warns when str.contains() gets a pattern with capture groups; we only need booleans.
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

# --------------------------------------------------------------------------------------
# Heuristic pattern sets. These are *diagnostics*, not labels; they are reported with
# their exact definitions so a reader can judge them.
# --------------------------------------------------------------------------------------
INDIAN_MARKERS = re.compile(
    r"\b(India|Indian|IPC|CrPC|Cr\.P\.C|BNS|BNSS|Bharatiya|Constitution of India|"
    r"Supreme Court of India|High Court|Lok Sabha|Rajya Sabha|Parliament of India|"
    r"Panchayat|lakh|crore|Rs\.?|₹|Union Territory|State Government|Article \d+)\b",
    re.IGNORECASE,
)
FOREIGN_MARKERS = re.compile(
    r"\b(United States|U\.S\.|USA|federal court|Supreme Court of the United States|"
    r"U\.S\.C\.|United Kingdom|England and Wales|House of Lords|EU law|GDPR|"
    r"feet|miles|\$\d)",
    re.IGNORECASE,
)
# Prompts that reference material the model never receives (orphaned context).
CONTEXT_DEPENDENT = re.compile(
    r"\b(in this context|the above|given (text|passage|context|excerpt)|this passage|"
    r"the passage|the following text|mentioned above|this document|the text above|"
    r"according to the (text|passage|document))\b",
    re.IGNORECASE,
)
# Multiple-choice / structured-answer signals (determines whether objective matching is possible).
MCQ_PATTERN = re.compile(r"(\(\s*[a-dA-D]\s*\)|\b[A-D]\)\s|\bOptions?\s*:)")
YES_NO_START = re.compile(r"^(is|are|can|does|do|did|was|were|will|should|may|has|have)\b", re.IGNORECASE)
ACT_PATTERN = re.compile(r"\b((?:[A-Z][A-Za-z()'&]*\s){1,8}(?:Act|Code|Sanhita|Adhiniyam)),?\s*(\d{4})")
ARTICLE_PATTERN = re.compile(r"\bArticle\s+(\d+[A-Z]?)\b")
SECTION_PATTERN = re.compile(r"\bSection\s+(\d+[A-Z]?)\b")
REASONING_LEAK_MARKERS = re.compile(r"(</?think>|final answer|\bAnswer:)", re.IGNORECASE)
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def normalize(text: str) -> str:
    """Case/whitespace/punctuation-insensitive normalization used for duplicate detection."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def describe_lengths(series: pd.Series) -> dict[str, float]:
    q = series.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "mean": round(float(series.mean()), 2),
        "std": round(float(series.std()), 2),
        **{f"p{int(k * 100)}": float(v) for k, v in q.items()},
    }


def near_duplicates(texts: list[str], threshold: float, chunk: int = 2000) -> dict[str, Any]:
    """Find each text's nearest neighbour by TF-IDF cosine similarity (word 1-2 grams).

    Exact (normalized) duplicates are included; the caller reports both separately.
    Returns counts at the threshold plus example pairs.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)  # rows are L2-normalized -> dot product == cosine
    n = X.shape[0]
    best_sim = np.zeros(n, dtype=np.float32)
    best_idx = np.full(n, -1, dtype=np.int64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = (X[start:end] @ X.T).toarray()
        sims[np.arange(end - start), np.arange(start, end)] = -1.0  # exclude self
        best_idx[start:end] = sims.argmax(axis=1)
        best_sim[start:end] = sims.max(axis=1)
    # Connected-component style grouping would be ideal; nearest-neighbour count is a
    # sufficient leakage signal: any row whose NN is above threshold could leak across splits.
    mask = best_sim >= threshold
    examples = []
    for i in np.where(mask)[0][:10]:
        j = int(best_idx[i])
        if normalize(texts[i]) != normalize(texts[j]):  # show non-identical pairs only
            examples.append({"i": int(i), "j": j, "cosine": round(float(best_sim[i]), 3),
                             "text_i": texts[i][:200], "text_j": texts[j][:200]})
    return {
        "threshold": threshold,
        "rows_with_neighbor_above_threshold": int(mask.sum()),
        "fraction": round(float(mask.mean()), 4),
        "nn_similarity_quantiles": {f"p{p}": round(float(np.percentile(best_sim, p)), 3)
                                    for p in (50, 75, 90, 95, 99)},
        "examples_non_identical": examples[:5],
        "_best_sim": best_sim,
    }


def audit(input_path: Path, out_dir: Path, nd_threshold: float) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_bytes = input_path.read_bytes()
    records = json.loads(raw_bytes.decode("utf-8"))
    log.info("Loaded %d records", len(records))

    report: dict[str, Any] = {
        "file": {
            "path": str(input_path).replace("\\", "/"),
            "size_bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "top_level_type": type(records).__name__,
        }
    }

    key_sets = Counter(tuple(sorted(r.keys())) for r in records)
    report["schema"] = {"key_sets": {"|".join(k): v for k, v in key_sets.items()}}
    df = pd.DataFrame(records)
    fields = list(df.columns)
    report["n_rows"] = len(df)
    report["columns"] = fields
    report["dtypes"] = {c: sorted({type(v).__name__ for v in df[c]}) for c in fields}

    # ---- missing / empty ----------------------------------------------------------
    missing = {}
    for c in fields:
        s = df[c]
        missing[c] = {
            "null": int(s.isna().sum()),
            "empty_or_whitespace": int(s.fillna("").astype(str).str.strip().eq("").sum()),
        }
    report["missing"] = missing
    for c in fields:
        df[c] = df[c].fillna("").astype(str)

    # ---- lengths ------------------------------------------------------------------
    lengths = {}
    for c in fields:
        words = df[c].str.split().str.len()
        lengths[c] = {"chars": describe_lengths(df[c].str.len()), "words": describe_lengths(words)}
        df[f"{c}_words"] = words
    report["lengths"] = lengths

    # ---- duplicates ---------------------------------------------------------------
    df["prompt_norm"] = df["prompt"].map(normalize)
    df["response_norm"] = df["response"].map(normalize)
    dup_prompt_groups = df.groupby("prompt_norm").size()
    conflicting = (
        df.groupby("prompt_norm")["response_norm"].nunique().loc[lambda s: s > 1]
    )
    report["duplicates"] = {
        "exact_full_row": int(df.duplicated(subset=fields).sum()),
        "exact_prompt": int(df.duplicated(subset=["prompt"]).sum()),
        "normalized_prompt": int(df.duplicated(subset=["prompt_norm"]).sum()),
        "normalized_prompt_and_response": int(df.duplicated(subset=["prompt_norm", "response_norm"]).sum()),
        "exact_response": int(df.duplicated(subset=["response"]).sum()),
        "prompt_groups_with_multiple_rows": int((dup_prompt_groups > 1).sum()),
        "largest_prompt_group": int(dup_prompt_groups.max()),
        "prompt_groups_with_conflicting_responses": int(len(conflicting)),
        "most_repeated_prompts": [
            {"prompt": p[:150], "count": int(n)}
            for p, n in dup_prompt_groups.sort_values(ascending=False).head(10).items()
        ],
    }

    log.info("Computing near-duplicate prompts (TF-IDF cosine >= %.2f)...", nd_threshold)
    nd = near_duplicates(df["prompt"].tolist(), nd_threshold)
    df["prompt_nn_sim"] = nd.pop("_best_sim")
    report["near_duplicates_prompt"] = nd

    # ---- task format --------------------------------------------------------------
    first_word = df["prompt"].str.strip().str.split().str[0].str.lower().str.strip("'\"*:,")
    report["task_format"] = {
        "prompts_with_mcq_pattern": int(df["prompt"].str.contains(MCQ_PATTERN).sum()),
        "prompts_ending_with_question_mark": int(df["prompt"].str.strip().str.endswith("?").sum()),
        "prompts_yes_no_style_start": int(df["prompt"].str.strip().str.match(YES_NO_START).sum()),
        "responses_starting_yes_or_no": int(
            df["response"].str.strip().str.match(r"^(yes|no)\b", case=False).sum()
        ),
        "prompt_first_word_top20": dict(Counter(first_word).most_common(20)),
        "prompts_context_dependent": int(df["prompt"].str.contains(CONTEXT_DEPENDENT).sum()),
        "context_dependent_examples": df.loc[
            df["prompt"].str.contains(CONTEXT_DEPENDENT), "prompt"
        ].head(8).str.slice(0, 200).tolist(),
    }

    # ---- reasoning field ----------------------------------------------------------
    cot_contains_response_prefix = (
        df.apply(lambda r: len(r["response"]) > 50 and r["response"][:80] in r["complex_cot"], axis=1)
    )
    report["reasoning_field"] = {
        "field": "complex_cot",
        "empty": int(df["complex_cot"].str.strip().eq("").sum()),
        "contains_think_or_answer_markers": int(df["complex_cot"].str.contains(REASONING_LEAK_MARKERS).sum()),
        "contains_first_80_chars_of_response": int(cot_contains_response_prefix.sum()),
        "note": "complex_cot is gold reasoning; it must never be given to the evaluated model at inference.",
    }

    # ---- domain / jurisdiction signals --------------------------------------------
    full_text = df["prompt"] + " " + df["response"]
    ind = full_text.str.contains(INDIAN_MARKERS)
    frn = full_text.str.contains(FOREIGN_MARKERS)
    report["domain_signals"] = {
        "definition": "regex heuristics over prompt+response; see INDIAN_MARKERS / FOREIGN_MARKERS in code",
        "rows_with_indian_marker": int(ind.sum()),
        "rows_with_foreign_marker": int(frn.sum()),
        "rows_with_foreign_but_no_indian_marker": int((frn & ~ind).sum()),
        "rows_with_neither": int((~frn & ~ind).sum()),
        "foreign_only_examples": df.loc[frn & ~ind, "prompt"].head(8).str.slice(0, 200).tolist(),
        "neither_examples": df.loc[~frn & ~ind, "prompt"].head(8).str.slice(0, 200).tolist(),
        "rows_with_devanagari": int(full_text.str.contains(DEVANAGARI).sum()),
        "non_ascii_char_fraction_prompt": round(
            float(df["prompt"].map(lambda t: sum(ord(ch) > 127 for ch in t) / max(len(t), 1)).mean()), 5
        ),
    }

    # ---- legal categories (none in the data; statute mentions as a proxy) ------------
    act_counter: Counter[str] = Counter()
    for t in df["prompt"]:
        for name, year in ACT_PATTERN.findall(t):
            act_counter[f"{name.strip()} {year}"] += 1
    art = df["prompt"].str.contains(ARTICLE_PATTERN)
    sec = df["prompt"].str.contains(SECTION_PATTERN)
    report["legal_categories"] = {
        "explicit_category_field": None,
        "prompts_mentioning_a_named_act_with_year": int(df["prompt"].str.contains(ACT_PATTERN).sum()),
        "prompts_mentioning_article": int(art.sum()),
        "prompts_mentioning_section": int(sec.sum()),
        "top_acts_in_prompts": dict(act_counter.most_common(30)),
        "distinct_acts_in_prompts": len(act_counter),
    }

    # ---- templating signal (city names appended to scenarios) ---------------------
    cities = ["Mumbai", "Delhi", "Chennai", "Kolkata", "Bengaluru", "Bangalore", "Hyderabad",
              "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Guwahati", "Kochi", "Chandigarh", "Bhopal"]
    city_re = re.compile(r"\b(" + "|".join(cities) + r")\b")
    report["templating_signal"] = {
        "prompts_mentioning_major_city": int(df["prompt"].str.contains(city_re).sum()),
        "city_counts": dict(Counter(m for t in df["prompt"] for m in city_re.findall(t)).most_common()),
    }

    # ---- split / labels -----------------------------------------------------------
    report["split_and_labels"] = {
        "predefined_split": "none (single file, no split column)",
        "label_columns": [],
        "difficulty_labels": "none",
        "task_type": "free-form question -> long-form answer generation (with reasoning trace)",
    }

    # ---- samples ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=6, replace=False)
    report["random_samples"] = [
        {"row": int(i), "prompt": df.at[i, "prompt"][:300], "response": df.at[i, "response"][:300]}
        for i in idx
    ]

    # Per-row audit flags are useful downstream (preprocessing reads them).
    flags = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "prompt_words": df["prompt_words"],
        "response_words": df["response_words"],
        "cot_words": df["complex_cot_words"],
        "is_dup_prompt_norm": df.duplicated(subset=["prompt_norm"], keep="first"),
        "prompt_nn_sim": df["prompt_nn_sim"].round(4),
        "context_dependent": df["prompt"].str.contains(CONTEXT_DEPENDENT),
        "indian_marker": ind,
        "foreign_marker": frn,
    })
    flags.to_csv(out_dir / "row_flags.csv", index=False)

    (out_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "dataset_audit.md").write_text(to_markdown(report), encoding="utf-8")
    log.info("Wrote %s", out_dir)
    return report


def to_markdown(r: dict[str, Any]) -> str:
    L = ["# Dataset audit: alpie/indian-law", ""]
    f = r["file"]
    L += [f"- File: `{f['path']}` ({f['size_bytes']:,} bytes, sha256 `{f['sha256'][:16]}…`)",
          f"- Rows: **{r['n_rows']:,}**", f"- Columns: {', '.join(r['columns'])}", ""]
    L += ["## Missing / empty", "| column | null | empty/whitespace |", "|---|---|---|"]
    L += [f"| {c} | {v['null']} | {v['empty_or_whitespace']} |" for c, v in r["missing"].items()]
    L += ["", "## Length (words)", "| column | mean | p5 | p50 | p95 | p99 | max |", "|---|---|---|---|---|---|---|"]
    for c, v in r["lengths"].items():
        w = v["words"]
        L.append(f"| {c} | {w['mean']} | {w['p5']:.0f} | {w['p50']:.0f} | {w['p95']:.0f} | {w['p99']:.0f} | {w['p100']:.0f} |")
    L += ["", "## Duplicates"] + [f"- {k}: {v}" for k, v in r["duplicates"].items() if k != "most_repeated_prompts"]
    L += ["", "Most repeated prompts:"] + [f"- ({d['count']}×) {d['prompt']}" for d in r["duplicates"]["most_repeated_prompts"]]
    nd = r["near_duplicates_prompt"]
    L += ["", f"## Near-duplicate prompts (TF-IDF cosine ≥ {nd['threshold']})",
          f"- Rows whose nearest neighbour is above threshold: {nd['rows_with_neighbor_above_threshold']} ({nd['fraction']:.2%})",
          f"- NN similarity quantiles: {nd['nn_similarity_quantiles']}"]
    for e in nd["examples_non_identical"]:
        L.append(f"  - {e['cosine']}: \"{e['text_i'][:120]}\" ↔ \"{e['text_j'][:120]}\"")
    L += ["", "## Task format"] + [f"- {k}: {v}" for k, v in r["task_format"].items() if not isinstance(v, (list, dict))]
    L += [f"- first words: {r['task_format']['prompt_first_word_top20']}", "", "Context-dependent examples:"]
    L += [f"- {e}" for e in r["task_format"]["context_dependent_examples"]]
    L += ["", "## Reasoning field"] + [f"- {k}: {v}" for k, v in r["reasoning_field"].items()]
    d = r["domain_signals"]
    L += ["", "## Domain / jurisdiction signals"] + [f"- {k}: {v}" for k, v in d.items() if not isinstance(v, list)]
    L += ["", "Foreign-only examples:"] + [f"- {e}" for e in d["foreign_only_examples"]]
    L += ["", "No-marker examples:"] + [f"- {e}" for e in d["neither_examples"]]
    lc = r["legal_categories"]
    L += ["", "## Legal categories", "No category field exists. Statute mentions in prompts (proxy):"]
    L += [f"- {k}: {v}" for k, v in lc.items() if k != "top_acts_in_prompts"]
    L += [f"  - {k}: {v}" for k, v in lc["top_acts_in_prompts"].items()]
    L += ["", "## Templating signal", f"- {r['templating_signal']}", "", "## Split / labels"]
    L += [f"- {k}: {v}" for k, v in r["split_and_labels"].items()]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/raw/Alpie-core_core_indian_law.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/audit"))
    ap.add_argument("--near-dup-threshold", type=float, default=0.9)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    audit(args.input, args.out_dir, args.near_dup_threshold)


if __name__ == "__main__":
    main()
