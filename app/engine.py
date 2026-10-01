"""Inference layer for the Streamlit demo.

Wraps the trained artefacts so the UI never touches torch directly:

* fine-tuned InLegalBERT Task A classifier (experiments/transformer/<run>/best.pt)
* adaptive test-time compute = Monte-Carlo-dropout passes routed by the deterministic
  pass's top-1 confidence, with the easy/medium/hard thresholds taken from the
  VALIDATION quantiles recorded in results/tt_compute_significance.json
* TF-IDF + kNN classical baseline (small, instant) and, optionally, TF-IDF + LR

Everything heavy is behind st.cache_resource / st.cache_data so a demo click costs
only the forward passes.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # the encoder's unused MLM head prints a long LOAD REPORT otherwise
# the baseline joblib was pickled with a newer scikit-learn; the transform path is unchanged
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.task_a import load_task_a  # noqa: E402
from src.training.train_transformer import load_trained  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

TRANSFORMER_RUN = "InLegalBERT_lr2e-05_do0.1_s42"
BUDGET = {"easy": 2, "medium": 8, "hard": 16}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Display names for the normalised act keys produced by src/data/provisions.py
ACT_DISPLAY = {
    "penal code": "Indian Penal Code",
    "code of criminal procedure": "Code of Criminal Procedure",
    "code of civil procedure": "Code of Civil Procedure",
    "bharatiya nyaya sanhita": "Bharatiya Nyaya Sanhita",
    "bharatiya nagarik suraksha sanhita": "Bharatiya Nagarik Suraksha Sanhita",
    "bharatiya sakshya adhiniyam": "Bharatiya Sakshya Adhiniyam",
    "negotiable instruments act": "Negotiable Instruments Act",
    "information technology act": "Information Technology Act",
    "right to information act": "Right to Information Act",
    "motor vehicles act": "Motor Vehicles Act",
    "companies act": "Companies Act",
    "evidence act": "Indian Evidence Act",
    "contract act": "Indian Contract Act",
    "hindu marriage act": "Hindu Marriage Act",
    "arbitration and conciliation act": "Arbitration and Conciliation Act",
    "consumer protection act": "Consumer Protection Act",
    "income tax act": "Income Tax Act",
}
SMALL_WORDS = {"of", "and", "the", "for", "on", "to", "in", "a", "an"}


def pretty_label(key: str) -> tuple[str, str]:
    """'art:21' -> ('Article 21', 'Constitution of India')."""
    if not isinstance(key, str) or not key:
        return "—", ""
    if key.startswith("art:"):
        return f"Article {key[4:]}", "Constitution of India"
    if key.startswith("sec:"):
        body = key[4:]
        num, _, act = body.partition("|")
        if act in ACT_DISPLAY:
            act_name = ACT_DISPLAY[act]
        else:
            act_name = " ".join(w if w in SMALL_WORDS else w.capitalize() for w in act.split())
        return f"Section {num}", act_name or "unresolved Act"
    return key, ""


# --------------------------------------------------------------------------- loaders
@st.cache_resource(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_resource(show_spinner=False)
def get_transformer():
    """Fine-tuned InLegalBERT classifier -> (model, tokenizer, label encoder, run config)."""
    run_dir = resolve(get_config()["paths"]["experiments"]) / "transformer" / TRANSFORMER_RUN
    model, tok, enc, rc = load_trained(run_dir, DEVICE)
    return model, tok, enc, rc


@st.cache_resource(show_spinner=False)
def get_baseline(name: str = "tfidf_knn"):
    import joblib

    return joblib.load(resolve(get_config()["paths"]["experiments"]) / "baselines" / name / "model.joblib")


@st.cache_data(show_spinner=False)
def get_thresholds() -> tuple[float, float]:
    p = resolve("results/tt_compute_significance.json")
    t = json.loads(p.read_text(encoding="utf-8"))["thresholds_from_val"]
    return float(t["conf_lo"]), float(t["conf_hi"])


@st.cache_data(show_spinner=False)
def get_task_a_frames() -> dict[str, pd.DataFrame]:
    return load_task_a(get_config())


@st.cache_data(show_spinner=False)
def get_examples(n: int = 8, seed: int = 7) -> pd.DataFrame:
    """Held-out TEST questions, so predictions can be checked against gold labels live."""
    test = get_task_a_frames()["test"]
    pool = test[(test["prompt_words"] >= 12) & (test["prompt_words"] <= 45)]
    pool = pool if len(pool) >= n else test
    return pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def gold_lookup() -> dict[str, tuple[str, str]]:
    """question text -> (gold label, tier) for every eligible Task A question."""
    q = pd.concat(get_task_a_frames().values())
    return {p.strip(): (lab, tier) for p, lab, tier in zip(q["prompt"], q["label"], q["tier"])}


@st.cache_data(show_spinner=False)
def label_support() -> dict[str, int]:
    """How many training questions carry each label (used to flag rare provisions)."""
    return get_task_a_frames()["train"]["label"].value_counts().to_dict()


@st.cache_data(show_spinner=False)
def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(resolve(rel))


@st.cache_data(show_spinner=False)
def read_json(rel: str) -> dict:
    return json.loads(resolve(rel).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- inference
@dataclass
class Ranked:
    label: str
    prob: float
    name: str = ""
    source: str = ""
    train_support: int = 0

    def __post_init__(self) -> None:
        self.name, self.source = pretty_label(self.label)


@dataclass
class Analysis:
    question: str
    bucket: str
    passes: int
    deterministic: list[Ranked]
    adaptive: list[Ranked]
    det_confidence: float
    adaptive_confidence: float
    entropy: float
    margin: float
    ms_total: float
    ms_per_pass: float
    gold: str | None = None
    gold_tier: str | None = None
    baseline: list[Ranked] = field(default_factory=list)
    baseline_ms: float = 0.0
    n_classes: int = 0

    @property
    def top(self) -> Ranked:
        return self.adaptive[0]

    @property
    def correct(self) -> bool | None:
        return None if self.gold is None else self.adaptive[0].label == self.gold

    @property
    def gold_rank(self) -> int | None:
        if self.gold is None:
            return None
        for i, r in enumerate(self.adaptive, 1):
            if r.label == self.gold:
                return i
        return None


def _set_dropout(model: torch.nn.Module, on: bool) -> None:
    model.eval()
    if on:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()


@torch.no_grad()
def _probs(model, ids, mask) -> torch.Tensor:
    """One forward pass in fp32.

    Training/evaluation used bf16 autocast for throughput; for a single question fp32 costs
    nothing and makes the "deterministic pass" genuinely deterministic, so the same question
    always routes to the same bucket in a live demo.
    """
    logits = model(ids, mask)
    return torch.softmax(logits.float(), dim=1).cpu()[0]


def _rank(probs: torch.Tensor, classes: np.ndarray, k: int = 5) -> list[Ranked]:
    support = label_support()
    order = torch.topk(probs, k=min(k, probs.numel())).indices.tolist()
    out = []
    for j in order:
        r = Ranked(str(classes[j]), float(probs[j]))
        r.train_support = int(support.get(r.label, 0))
        out.append(r)
    return out


def analyse(question: str, strategy: str = "adaptive", with_baseline: bool = True,
            baseline_name: str = "tfidf_knn", seed: int = 1000) -> Analysis:
    """One deterministic pass + MC-dropout passes according to the chosen compute strategy."""
    question = question.strip()
    model, tok, enc, rc = get_transformer()
    classes = np.array(enc.classes_, dtype=object)
    batch = tok([question], truncation=True, max_length=rc.max_length, return_tensors="pt")
    ids, mask = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)

    t0 = time.perf_counter()
    _set_dropout(model, False)
    det = _probs(model, ids, mask)
    conf = float(det.max())

    lo, hi = get_thresholds()
    if strategy == "adaptive":
        bucket = "easy" if conf >= hi else ("medium" if conf >= lo else "hard")
        n_passes = BUDGET[bucket]
    elif strategy == "single":
        bucket, n_passes = "fixed", 1
    else:  # fixed_4 / fixed_8 / fixed_16
        n_passes = int(strategy.split("_")[1])
        bucket = "fixed"

    acc = det.clone()
    if n_passes > 1:
        set_seed(seed)
        _set_dropout(model, True)
        for _ in range(n_passes - 1):
            acc += _probs(model, ids, mask)
        _set_dropout(model, False)
    mean = acc / n_passes
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms_total = 1000 * (time.perf_counter() - t0)

    srt = torch.sort(mean, descending=True).values
    entropy = float(-(mean * torch.log(mean.clamp(min=1e-12))).sum())
    margin = float(srt[0] - srt[1]) if mean.numel() > 1 else 1.0

    gold_tier = gold_lookup().get(question)
    base: list[Ranked] = []
    base_ms = 0.0
    if with_baseline:
        try:
            clf = get_baseline(baseline_name)
            t1 = time.perf_counter()
            labels, scores = clf.rank([question], k=5)
            base_ms = 1000 * (time.perf_counter() - t1)
            support = label_support()
            for lab, sc in zip(labels[0], scores[0]):
                r = Ranked(str(lab), float(sc))
                r.train_support = int(support.get(r.label, 0))
                base.append(r)
        except Exception:  # a missing joblib must not break the live demo
            base, base_ms = [], 0.0

    return Analysis(
        question=question, bucket=bucket, passes=n_passes,
        deterministic=_rank(det, classes), adaptive=_rank(mean, classes),
        det_confidence=conf, adaptive_confidence=float(mean.max()),
        entropy=entropy, margin=margin,
        ms_total=ms_total, ms_per_pass=ms_total / n_passes,
        gold=gold_tier[0] if gold_tier else None,
        gold_tier=gold_tier[1] if gold_tier else None,
        baseline=base, baseline_ms=base_ms, n_classes=len(enc),
    )


@st.cache_data(show_spinner=False)
def quick_eval(n: int = 40, tier: str = "all", seed: int = 0) -> pd.DataFrame:
    """Score n held-out TEST questions in one batched deterministic pass.

    This is the same setting as the `single` row of results/tt_compute_summary.csv, so the
    accuracy shown live is directly comparable to the reported test accuracy.
    """
    model, tok, enc, rc = get_transformer()
    classes = np.array(enc.classes_, dtype=object)
    df = get_task_a_frames()["test"]
    if tier != "all":
        df = df[df["tier"] == tier]
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    batch = tok(df["prompt"].tolist(), truncation=True, max_length=rc.max_length,
                padding=True, return_tensors="pt")
    _set_dropout(model, False)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
    probs = torch.softmax(logits.float(), dim=1).cpu()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / len(df)

    top = torch.topk(probs, k=5, dim=1)
    rows = []
    for i, r in df.iterrows():
        idx = top.indices[i].tolist()
        preds = [str(classes[j]) for j in idx]
        rows.append({
            "question": r["prompt"], "tier": r["tier"],
            "gold": r["label"], "gold_name": " · ".join(pretty_label(r["label"])),
            "pred": preds[0], "pred_name": " · ".join(pretty_label(preds[0])),
            "confidence": float(top.values[i][0]),
            "correct": preds[0] == r["label"],
            "in_top5": r["label"] in preds,
            "ms": ms,
        })
    return pd.DataFrame(rows)


def warm_up() -> dict[str, float]:
    """Load + one tiny forward pass so the first user click is already warm."""
    t0 = time.perf_counter()
    get_transformer()
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    analyse("Which provision guarantees the right to life?", strategy="single", with_baseline=False)
    return {"load_seconds": load_s, "first_call_seconds": time.perf_counter() - t1, "device": DEVICE}
