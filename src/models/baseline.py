"""Classical ML baselines for Task A: question text -> provision label.

* TF-IDF + Logistic Regression (one-vs-rest, liblinear). Multinomial LR over ~1.9k classes x
  50k features did not finish in 10 min on this machine; OvR fits in ~100 s and is the same
  model family.
* TF-IDF + similarity-weighted k-nearest-neighbour vote (retrieval baseline).

Random Forest is intentionally NOT used: with ~1.9k classes each tree stores a class
distribution per node (~200 MB/tree at this data size), which does not fit in 24 GB RAM for
a useful number of trees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion


def build_vectorizer(kind: str, max_features: int) -> Any:
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=max_features)
    if kind == "word":
        return word
    if kind == "word_char":
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=max_features)
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"unknown feature kind {kind!r}")


@dataclass
class TfidfClassifier:
    """Common interface: fit(texts, labels); rank(texts, k) -> (labels[k], scores[k]) per text."""

    model: str
    params: dict[str, Any]
    max_features: int = 50000
    vectorizer: Any = field(init=False, default=None)
    clf: Any = field(init=False, default=None)
    classes_: np.ndarray = field(init=False, default=None)
    _X: sp.csr_matrix = field(init=False, default=None)
    _y: np.ndarray = field(init=False, default=None)

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfClassifier":
        self.vectorizer = build_vectorizer(self.params.get("features", "word"), self.max_features)
        X = sp.csr_matrix(self.vectorizer.fit_transform(texts))
        X.sort_indices()  # joblib workers get read-only memmaps; unsorted indices then crash liblinear
        y = np.asarray(labels, dtype=object)
        if self.model == "tfidf_lr":
            self.clf = OneVsRestClassifier(
                LogisticRegression(C=self.params["C"], solver="liblinear"), n_jobs=-1
            ).fit(X, y)
            self.classes_ = self.clf.classes_
        elif self.model == "tfidf_knn":
            # L2-normalise rows so dot product == cosine (FeatureUnion output is not normalised).
            norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
            self._X = sp.diags(1 / np.maximum(norms, 1e-12)) @ X
            self._y = y
            self.classes_ = np.array(sorted(set(labels)), dtype=object)
        else:
            raise ValueError(self.model)
        return self

    def _scores(self, texts: list[str]) -> np.ndarray:
        X = sp.csr_matrix(self.vectorizer.transform(texts))
        if self.model == "tfidf_lr":
            return self.clf.predict_proba(X)
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
        X = sp.diags(1 / np.maximum(norms, 1e-12)) @ X
        sims = (X @ self._X.T).toarray()
        k = self.params["k"]
        col = {c: i for i, c in enumerate(self.classes_)}
        label_idx = np.array([col[v] for v in self._y])
        scores = np.zeros((len(texts), len(self.classes_)))
        top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        for i, nbrs in enumerate(top):
            np.add.at(scores[i], label_idx[nbrs], sims[i, nbrs])
        return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)

    def rank(self, texts: list[str], k: int = 5) -> tuple[list[list[str]], np.ndarray]:
        scores = self._scores(texts)
        order = np.argsort(-scores, axis=1)[:, :k]
        labels = [[self.classes_[j] for j in row] for row in order]
        return labels, np.take_along_axis(scores, order, axis=1)
