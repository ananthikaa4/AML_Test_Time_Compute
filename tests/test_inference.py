"""Tests for prompt construction, SFT target truncation and answer extraction for Model A."""
from __future__ import annotations

from src.data.provisions import first_provision
from src.models.reasoning_model import SYSTEM_PROMPT, messages, truncate_answer


def test_prompt_contains_only_question():
    msgs = messages("What does Article 21 protect?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "What does Article 21 protect?"
    assert msgs[0]["content"] == SYSTEM_PROMPT  # no reference answer / reasoning in the prompt


def test_truncate_answer_keeps_whole_sentences():
    text = "First sentence here. Second one is here too. Third sentence is long " + "word " * 50 + "."
    out = truncate_answer(text, max_words=10)
    assert out == "First sentence here. Second one is here too."


def test_truncate_answer_always_keeps_one_sentence_capped():
    out = truncate_answer("word " * 300 + ".", max_words=20)
    assert len(out.split()) == 20


def test_truncate_answer_strips_markdown():
    out = truncate_answer("**Answer:** Under ## Article 21, life is protected.", max_words=50)
    assert "*" not in out and "#" not in out and out.startswith("Answer:")


def test_truncation_preserves_leading_citation():
    ref = "The quorum is one-tenth of members under Article 189(3) of the Constitution. " + "More text. " * 40
    q = "What is the quorum of a State Legislature?"
    assert first_provision(truncate_answer(ref, 120), q) == first_provision(ref, q) == "art:189"
