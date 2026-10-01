"""Heuristic legal-area assignment.

The dataset has NO category field. For the per-category analysis (RQ4) we derive a coarse
legal area from keywords in the *question only* (never the answer, to keep it usable at
inference time). This is a documented heuristic, not ground truth; the first matching
area in `AREAS` order wins, and unmatched questions are "general/other".
"""
from __future__ import annotations

import re

AREAS: dict[str, list[str]] = {
    "constitutional": [r"\bconstitution", r"\barticle\s+\d", r"\bfundamental right", r"\bparliament\b",
                       r"\bpresident\b", r"\bgovernor\b", r"\blegislature\b", r"\bpanchayat", r"\bmunicipalit",
                       r"\bschedule\b", r"\bamendment\) act", r"\bunion territor", r"\bwrit\b", r"\belection commission"],
    "criminal": [r"\bipc\b", r"\bpenal code", r"\bcrpc\b", r"\bcriminal", r"\bbail\b", r"\bfir\b", r"\bmurder",
                 r"\btheft\b", r"\bcheating\b", r"\bnyaya sanhita", r"\bnagarik suraksha", r"\bndps\b", r"\boffence"],
    "evidence_procedure": [r"\bevidence act", r"\bcode of civil procedure", r"\bcpc\b", r"\blimitation act",
                           r"\bsakshya", r"\bdecree\b", r"\bpleading", r"\bsuit\b", r"\border\s+[xvi]+\b"],
    "contract_commercial": [r"\bcontract", r"\bpartnership", r"\bsale of goods", r"\bnegotiable instrument",
                            r"\bcheque", r"\barbitration", r"\bspecific relief", r"\bagency\b", r"\bguarantee"],
    "corporate_finance_tax": [r"\bcompan(y|ies)", r"\binsolvency", r"\bsebi\b", r"\bincome tax", r"\bgst\b",
                              r"\btax", r"\bbank", r"\bfema\b", r"\bcompetition act"],
    "property_land": [r"\btransfer of property", r"\bmortgage", r"\blease\b", r"\bland acquisition", r"\brfctlarr",
                      r"\beasement", r"\bregistration act", r"\btrust", r"\bproperty\b", r"\brera\b"],
    "family_personal": [r"\bmarriage", r"\bdivorce", r"\bsuccession", r"\bhindu\b", r"\bmuslim", r"\bmaintenance\b",
                        r"\badoption", r"\bguardian", r"\bdowry", r"\bdomestic violence", r"\bwill\b"],
    "labour_employment": [r"\bindustrial dispute", r"\bworkm[ae]n", r"\bfactories act", r"\bwages?\b",
                          r"\bemployee", r"\bemployer", r"\blabou?r", r"\bgratuity", r"\bprovident fund"],
    "ip_technology": [r"\bcopyright", r"\bpatent", r"\btrademark", r"\bdesigns act", r"\bit act\b",
                      r"\binformation technology", r"\bcyber", r"\bdata protection", r"\bintellectual property"],
    "consumer_regulatory": [r"\bconsumer", r"\bmotor vehicle", r"\benvironment", r"\bpollution", r"\bforest",
                            r"\bwildlife", r"\bbiodiversity", r"\bright to information", r"\brti\b", r"\bfood safety"],
}
_COMPILED = {area: re.compile("|".join(pats), re.IGNORECASE) for area, pats in AREAS.items()}
OTHER = "general_other"


def legal_area(question: str) -> str:
    for area, rx in _COMPILED.items():
        if rx.search(question):
            return area
    return OTHER
