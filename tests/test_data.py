"""Tests for cleaning, provision extraction, Task A labelling and leakage-safe splitting."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.categories import OTHER, legal_area
from src.data.preprocess import clean_text, is_orphan, normalize_prompt, task_a_label
from src.data.provisions import (
    act_key,
    extract_citations,
    first_provision,
    is_correct,
    label_number_in_text,
    provisions_in_question,
)
from src.data.split import UnionFind, assign_splits, near_duplicate_groups


# ---------------------------------------------------------------- cleaning
def test_clean_text_normalises_whitespace_and_unicode():
    assert clean_text("  Article  21 \t\n\n\n\nof  the ﬁle ") == "Article 21\n\nof the file"


def test_clean_text_removes_control_chars():
    assert clean_text("a\x00b\x07c") == "abc"


def test_normalize_prompt_is_case_and_punctuation_insensitive():
    assert normalize_prompt("What is Article 21?") == normalize_prompt("what is article 21")


@pytest.mark.parametrize("prompt,expected", [
    ("What does definition refer to in this context?", True),
    ("How do these provisions align with national goals?", True),
    ("How does Article 243D reserve seats? Discuss these provisions.", False),  # names the law itself
    ("A agrees to supply B with necessary articles under the Indian Contract Act. "
     "What constitutes necessary articles in this context?", False),  # long, self-contained scenario
])
def test_orphan_detection(prompt, expected):
    assert is_orphan(prompt) is expected


# ---------------------------------------------------------------- provisions
def test_article_with_clause_is_reduced_to_article():
    assert first_provision("This is per Article 189(3) of the Constitution.", "q") == "art:189"


def test_article_suffix_letter_kept():
    assert first_provision("Governed by Article 243G.", "q") == "art:243G"


def test_international_instrument_articles_ignored():
    assert first_provision("Article 17 of the ICCPR protects privacy; in India, Article 21 applies.", "q") == "art:21"


def test_section_resolves_act_after_it_and_aliases():
    assert first_provision("Under Section 302 of the IPC, murder...", "q") == "sec:302|penal code"
    assert first_provision("Section 302 of the Indian Penal Code, 1860 ...", "q") == "sec:302|penal code"


def test_section_falls_back_to_single_act_in_question():
    q = "How does the Companies Act, 2013 regulate buybacks?"
    assert first_provision("Section 68 of the Act permits it.", q) == "sec:68|companies act"


def test_unresolvable_section_ignored():
    assert first_provision("Section 10 applies.", "What does it say?") is None


def test_section_keyword_case_insensitive():
    assert provisions_in_question("Explain the significance of 'section 27' of the Indian Evidence Act") == {"sec:27"}


def test_act_key_normalisation():
    assert act_key("the Indian Contract Act, 1872") == "contract act"
    assert act_key("IPC") == "penal code"


def test_label_number_in_text_catches_bare_numbers():
    assert label_number_in_text("sec:354|penal code", "Can a man be a victim under IPC 354?")
    assert not label_number_in_text("art:21", "What does Article 21A guarantee?")


def test_is_correct_uses_first_citation_only():
    q = "Which provision guarantees the right to life?"
    assert is_correct("Article 21 guarantees it; see also Article 14.", "art:21", q)
    assert not is_correct("Article 14 and Article 21 are relevant.", "art:21", q)


def test_extract_citations_ordered_by_position():
    cites = extract_citations("Section 5 of the Limitation Act and Article 14 apply.")
    assert [c.key for c in cites] == ["sec:5|limitation act", "art:14"]


# ---------------------------------------------------------------- Task A labels
def test_task_a_majority_label_and_agreement():
    refs = ["Per Article 21 ...", "Article 21 says ...", "Article 14 says ..."]
    out = task_a_label("Which right protects life?", refs, min_agreement=0.5)
    assert out["task_a_label"] == "art:21" and out["task_a_eligible"]
    assert out["task_a_agreement"] == pytest.approx(0.667, abs=1e-3)


def test_task_a_rejects_tie_and_copy():
    tie = task_a_label("Which right?", ["Article 21 ...", "Article 14 ..."], 0.5)
    assert not tie["task_a_eligible"] and tie["task_a_reason"] == "tied_labels"
    copy = task_a_label("What does Article 21 say?", ["Article 21 says ..."], 0.5)
    assert not copy["task_a_eligible"] and copy["task_a_reason"] == "label_in_question"


def test_task_a_no_citation():
    assert task_a_label("What is a tort?", ["A tort is a civil wrong."], 0.5)["task_a_reason"] == "no_citation"


# ---------------------------------------------------------------- categories
def test_legal_area():
    assert legal_area("What does Article 21 of the Constitution guarantee?") == "constitutional"
    assert legal_area("What is the punishment for theft under the IPC?") == "criminal"
    assert legal_area("What is jurisprudence?") == OTHER


# ---------------------------------------------------------------- splitting
def test_union_find_transitive():
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2) != uf.find(3)


def test_near_duplicates_share_group():
    texts = ["What is the role of the Registrar of Firms in India?",
             "Explain the role of the Registrar of Firms in India.",
             "What is the punishment for murder under the Penal Code?",
             "What is the punishment for murder under the Penal Code in India?",
             "How are Rajya Sabha members elected?"]
    g = near_duplicate_groups(texts * 2, threshold=0.8)  # x2 so min_df=2 keeps all terms
    assert g[2] == g[3]
    assert g[0] != g[4]


def test_assign_splits_keeps_groups_whole_and_is_deterministic():
    groups = pd.DataFrame({"group_id": range(200), "stratum": [i % 2 for i in range(200)], "n_questions": [1, 3] * 100})
    a = assign_splits(groups, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    b = assign_splits(groups, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    assert a == b
    assert set(a) == set(range(200))  # every group assigned exactly once
    per_split = pd.Series(a).map({"train": 0, "val": 1, "test": 2}).groupby(lambda gid: a[gid]).size()
    assert per_split["test"] > 0 and per_split["val"] > 0
