"""Tests for label encoding, classical baselines and Transformer pooling / optimiser setup."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.data.task_a import LabelEncoder
from src.models.baseline import TfidfClassifier
from src.models.transformer import TransformerClassifier

TEXTS = ["right to life and personal liberty", "freedom of speech and expression",
         "equality before law", "right to life dignity", "speech expression press freedom",
         "equal protection of laws"] * 3
LABELS = ["art:21", "art:19", "art:14", "art:21", "art:19", "art:14"] * 3


def test_label_encoder_roundtrip_and_unseen():
    enc = LabelEncoder(["b", "a", "b"])
    assert enc.classes_ == ["a", "b"] and len(enc) == 2
    assert enc.encode(["a", "b", "zzz"]) == [0, 1, -1]
    assert enc.decode([1, 0]) == ["b", "a"]


@pytest.mark.parametrize("model,params", [
    ("tfidf_lr", {"features": "word", "C": 10}),
    ("tfidf_lr", {"features": "word_char", "C": 10}),
    ("tfidf_knn", {"features": "word", "k": 3}),
])
def test_baselines_fit_and_rank(model, params):
    clf = TfidfClassifier(model, params, max_features=1000).fit(TEXTS, LABELS)
    labels, scores = clf.rank(["right to life", "freedom of the press"], k=2)
    assert labels[0][0] == "art:21" and labels[1][0] == "art:19"
    assert scores.shape == (2, 2) and (scores[:, 0] >= scores[:, 1]).all()


def _bare_classifier(pooling: str) -> TransformerClassifier:
    """Instance without downloading weights: only the pooling logic is exercised."""
    m = TransformerClassifier.__new__(TransformerClassifier)
    nn.Module.__init__(m)
    m.pooling = pooling
    return m


def test_mean_pooling_ignores_padding():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(_bare_classifier("mean").pool(h, mask), torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(_bare_classifier("cls").pool(h, mask), torch.tensor([[1.0, 1.0]]))
