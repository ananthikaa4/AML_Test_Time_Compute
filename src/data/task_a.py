"""Task A (provision identification) data access.

Model input is the question text ONLY. The label comes from the reference answers and is
never shown to a model; `complex_cot` and the references are not part of these frames.
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import resolve

SPLITS = ("train", "val", "test")
COLUMNS = ["qid", "prompt", "task_a_label", "task_a_tier", "legal_area", "prompt_words", "split"]


def load_questions(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(resolve(cfg["data"]["processed_dir"]) / "questions.parquet")


def load_task_a(cfg: dict) -> dict[str, pd.DataFrame]:
    q = load_questions(cfg)
    q = q[q["task_a_eligible"]][COLUMNS].rename(columns={"task_a_label": "label", "task_a_tier": "tier"})
    return {s: q[q["split"] == s].reset_index(drop=True) for s in SPLITS}


class LabelEncoder:
    """Maps train labels to ids. Labels unseen in training map to -1 (always counted wrong)."""

    def __init__(self, labels: list[str]) -> None:
        self.classes_ = sorted(set(labels))
        self.index = {c: i for i, c in enumerate(self.classes_)}

    def encode(self, labels: list[str]) -> list[int]:
        return [self.index.get(label, -1) for label in labels]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.classes_[i] for i in ids]

    def __len__(self) -> int:
        return len(self.classes_)
