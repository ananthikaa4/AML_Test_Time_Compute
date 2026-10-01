"""Leakage-safe grouped splitting.

Questions are grouped into near-duplicate components (TF-IDF cosine >= threshold between
normalised prompts, closed transitively with union-find). A whole component always lands
in one split, so a test question never has a paraphrase in train.

Components are assigned to train/val/test by seeded shuffling within strata, so the Task A
(provision-identification) subset is represented proportionally in every split.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_groups(texts: list[str], threshold: float, chunk: int = 2000) -> np.ndarray:
    """Component id for each text; texts whose TF-IDF cosine >= threshold share a component."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    n = X.shape[0]
    uf = UnionFind(n)
    n_pairs = 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = (X[start:end] @ X.T).tocoo()
        keep = (sims.data >= threshold) & (sims.row + start < sims.col)  # upper triangle, no self
        for r, c in zip(sims.row[keep] + start, sims.col[keep]):
            uf.union(int(r), int(c))
        n_pairs += int(keep.sum())
    roots = np.array([uf.find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    sizes = np.bincount(comp)
    log.info("near-dup pairs >= %.2f: %d; components: %d; largest: %d; singleton: %d",
             threshold, n_pairs, len(sizes), sizes.max(), int((sizes == 1).sum()))
    return comp


def assign_splits(groups: pd.DataFrame, ratios: dict[str, float], seed: int) -> dict[int, str]:
    """groups: one row per component with columns [group_id, stratum, n_questions].

    Within each stratum, components are shuffled and greedily filled into test, then val,
    until each reaches its question-count target; the rest go to train. Oversized components
    (> half of a target) always go to train so they cannot swamp a small split.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, str] = {}
    for _, g in groups.groupby("stratum"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        total = g["n_questions"].sum()
        targets = {"test": ratios["test"] * total, "val": ratios["val"] * total}
        filled = {"test": 0, "val": 0}
        for gid, n in zip(g["group_id"], g["n_questions"]):
            dest = "train"
            for split in ("test", "val"):
                if n <= 0.5 * targets[split] and filled[split] + n <= targets[split]:
                    dest = split
                    filled[split] += n
                    break
            out[int(gid)] = dest
    return out
