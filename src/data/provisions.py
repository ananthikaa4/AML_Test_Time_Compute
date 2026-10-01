"""Extraction of cited legal provisions (Constitution Articles, statute Sections).

This module defines Task A ("provision identification"): the gold label of a question is
the *first* provision cited in its reference answer, and a model answer is correct iff the
first provision it cites is the same. The *same* extractor is applied to gold answers and
to model outputs, so the scoring rule is symmetric.

Label keys:
    "art:<N>"             Constitution of India, Article N (clause suffixes like (3) dropped)
    "sec:<N>|<act_key>"   Section N of a named statute (act normalised by `act_key`)

Sections are only labelled when the Act can be resolved: either named right after the
section ("Section 36 of the Arbitration and Conciliation Act") or, failing that, when the
*question* names exactly one Act. Unresolvable sections are ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# "Article 243G", "Article 189(3)", "Articles 14 and 21" -> first number only per match.
# Keyword is case-insensitive ("section 27"); the suffix letter stays upper-case only ("21A").
ARTICLE_RE = re.compile(r"\b(?i:art(?:icle|\.)s?)\s+(\d{1,3}[A-Z]{0,3})\b")
SECTION_RE = re.compile(r"\b(?:(?i:sections?|sec\.)|S\.)\s+(\d{1,4}[A-Z]{0,3})\b")
# Articles of non-constitutional instruments must not be read as Constitution articles.
NON_CONSTITUTION_CONTEXT = re.compile(
    r"^[^.;\n]{0,40}\b(Convention|Covenant|UDHR|ICCPR|ICESCR|Declaration|Treaty|Charter|"
    r"GDPR|Model Law|Statute of|Agreement|TRIPS|GATT|UNCITRAL|Protocol)\b",
    re.IGNORECASE,
)
# "... Act, 1996" / "... Code, 1908" / "... Sanhita, 2023"
ACT_NAME_RE = re.compile(
    r"((?:[A-Z][\w()'&.-]*\s+(?:of\s+|and\s+|for\s+|the\s+|to\s+|on\s+)*){0,10}"
    r"(?:Act|Code|Sanhita|Adhiniyam))\b(?:,?\s*(\d{4}))?"
)
ACT_ALIASES = {
    "ipc": "penal code",
    "crpc": "code of criminal procedure",
    "cr.p.c": "code of criminal procedure",
    "cpc": "code of civil procedure",
    "bns": "bharatiya nyaya sanhita",
    "bnss": "bharatiya nagarik suraksha sanhita",
    "bsa": "bharatiya sakshya adhiniyam",
    "ni act": "negotiable instruments act",
    "it act": "information technology act",
    "rti act": "right to information act",
    "mv act": "motor vehicles act",
}
ALIAS_RE = re.compile(r"\b(IPC|CrPC|Cr\.P\.C|CPC|BNS|BNSS|BSA|NI Act|IT Act|RTI Act|MV Act)\b")
# Window after a section number in which we look for its Act.
SECTION_ACT_WINDOW = 90


def act_key(name: str) -> str:
    """Normalise an Act name: lowercase, drop year, leading article/"indian", punctuation."""
    s = name.lower().strip()
    s = ACT_ALIASES.get(s, s)
    s = re.sub(r",?\s*\d{4}$", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"^(the|under|of|in|and)\s+", "", s.strip())
    s = re.sub(r"^(the\s+)?indian\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = ACT_ALIASES.get(s, s)
    # "penal code" is the IPC regardless of "Indian" prefix
    return s


def find_acts(text: str) -> list[tuple[int, str]]:
    """Return (position, act_key) for each Act mentioned in `text`, in order."""
    out: list[tuple[int, str]] = []
    for m in ACT_NAME_RE.finditer(text):
        key = act_key(m.group(1))
        # Reject fragments such as "Act" alone or "This Act" / "The said Act".
        words = key.split()
        if len(words) < 2 or words[0] in {"this", "said", "same", "such", "an", "a", "any", "which", "that"}:
            continue
        out.append((m.start(), key))
    for m in ALIAS_RE.finditer(text):
        out.append((m.start(), act_key(m.group(1))))
    return sorted(out)


@dataclass(frozen=True)
class Citation:
    pos: int
    key: str


def extract_citations(text: str, fallback_act: str | None = None) -> list[Citation]:
    """All resolvable provision citations in `text`, ordered by position."""
    cites: list[Citation] = []
    for m in ARTICLE_RE.finditer(text):
        tail = text[m.end(): m.end() + 60]
        if NON_CONSTITUTION_CONTEXT.match(tail):
            continue
        cites.append(Citation(m.start(), f"art:{m.group(1).upper()}"))
    acts = find_acts(text)
    for m in SECTION_RE.finditer(text):
        act = None
        for pos, key in acts:
            if m.end() <= pos <= m.end() + SECTION_ACT_WINDOW:
                act = key
                break
        act = act or fallback_act
        if act:
            cites.append(Citation(m.start(), f"sec:{m.group(1).upper()}|{act}"))
    return sorted(cites, key=lambda c: c.pos)


def single_act(text: str) -> str | None:
    """The Act named in `text` if exactly one distinct Act is named, else None."""
    keys = {k for _, k in find_acts(text)}
    return next(iter(keys)) if len(keys) == 1 else None


def provision_number(key: str) -> str:
    """'art:21' -> 'art:21'; 'sec:36|arbitration...' -> 'sec:36' (for leakage checks)."""
    return key.split("|")[0]


def first_provision(answer: str, question: str) -> str | None:
    """Primary (first-cited) provision of `answer`, resolving bare sections via the question's Act.

    This is the scoring rule for Task A, applied identically to gold and predicted answers.
    """
    cites = extract_citations(answer, fallback_act=single_act(question))
    return cites[0].key if cites else None


def provisions_in_question(question: str) -> set[str]:
    """Provision numbers already given in the question (a label found here would be a copy)."""
    return {provision_number(c.key) for c in extract_citations(question, fallback_act=single_act(question) or "_")}


def label_number_in_text(key: str, text: str) -> bool:
    """True if the provision number of `key` appears as a standalone token in `text`.

    Stricter than `provisions_in_question`: also catches "IPC 354", "u/s 138", "Art 21" etc.
    """
    number = provision_number(key).split(":")[1]
    return re.search(rf"(?<![\w.]){re.escape(number)}(?![\w])", text, re.IGNORECASE) is not None


def is_correct(pred_answer: str, gold_key: str, question: str) -> bool:
    return first_provision(pred_answer, question) == gold_key
