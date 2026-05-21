"""Deterministic scorers for NeuroLit evaluation.

Day 4 part 1 builds only the deterministic metrics - the ones with an
unambiguous right answer that need no LLM judgment:

  - citation extraction: pull PMIDs out of the agent's answer text
  - PMID resolution:      does each cited PMID resolve to a real PubMed paper?
  - hallucination rate:   fraction of cited PMIDs that do NOT resolve
  - citation precision:   fraction of cited PMIDs that DO resolve
  - recall (lenient):     fraction of gold PMIDs that appear in the answer
  - adversarial pass:     for no-paper questions, did the agent correctly
                          avoid producing citations?

LLM-as-judge metrics (relevance accuracy, description faithfulness) are
deliberately deferred to Day 4 part 2.
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_TIMEOUT = 15

# PMIDs are runs of digits, typically 7-8 long. We match "PMID: 12345678"
# and "PMID 12345678" and bare "pmid=12345678" forms.
_PMID_PATTERN = re.compile(r"PMID[:\s=]*(\d{4,9})", re.IGNORECASE)


def extract_pmids(answer_text: str) -> list[str]:
    """Pull all cited PMIDs out of an agent answer, de-duplicated, in order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _PMID_PATTERN.finditer(answer_text or ""):
        pmid = match.group(1)
        if pmid not in seen:
            seen.add(pmid)
            ordered.append(pmid)
    return ordered


def pmid_resolves(pmid: str) -> bool:
    """True if the PMID resolves to a real record in PubMed.

    Uses esummary, which is lightweight. A real PMID returns a result block
    keyed by that PMID; a fake one returns an empty/absent result.
    """
    try:
        resp = requests.get(
            f"{NCBI_BASE}/esummary.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {})
        # esummary echoes requested uids; a valid record has no 'error' key.
        entry = result.get(pmid)
        if not entry:
            return False
        return "error" not in entry
    except (requests.RequestException, ValueError):
        # On a network/parse error we can't confirm - treat as unresolved
        # but the caller logs this distinctly so it's not silently a "hallucination".
        return False


def score_citations(answer_text: str, gold_pmids: list[str]) -> dict[str, Any]:
    """Score the citation-level deterministic metrics for one answer.

    Returns a dict with cited PMIDs, which resolved, precision, hallucination
    rate, and lenient recall against the gold set.
    """
    cited = extract_pmids(answer_text)

    resolved: list[str] = []
    unresolved: list[str] = []
    for pmid in cited:
        if pmid_resolves(pmid):
            resolved.append(pmid)
        else:
            unresolved.append(pmid)
        time.sleep(0.34)  # Stay under NCBI's ~3 req/sec unauthenticated limit.

    n_cited = len(cited)
    precision = len(resolved) / n_cited if n_cited else None
    hallucination_rate = len(unresolved) / n_cited if n_cited else None

    if gold_pmids:
        found_gold = [p for p in gold_pmids if p in cited]
        recall = len(found_gold) / len(gold_pmids)
    else:
        found_gold = []
        recall = None

    return {
        "cited_pmids": cited,
        "n_cited": n_cited,
        "resolved_pmids": resolved,
        "unresolved_pmids": unresolved,
        "citation_precision": precision,
        "hallucination_rate": hallucination_rate,
        "gold_pmids": gold_pmids,
        "gold_found": found_gold,
        "recall": recall,
    }


def score_adversarial(answer_text: str) -> dict[str, Any]:
    """Score an adversarial (no-paper-exists) question.

    A correct answer cites NO papers AND is a real response (not an error).
    We also do a light text check for an explicit not-found statement.
    """
    text = (answer_text or "")
    cited = extract_pmids(text)

    # A crashed run also produces zero citations - don't score that as a pass.
    is_error = text.strip().startswith("(eval error:")

    lower = text.lower()
    not_found_phrases = (
        "no such paper", "could not be found", "not found", "does not exist",
        "no paper", "unable to find", "no 2024 paper", "no 2023 paper",
    )
    states_not_found = any(phrase in lower for phrase in not_found_phrases)

    # Pass = a real response that produced no citations.
    passed = (not is_error) and len(cited) == 0

    return {
        "cited_pmids": cited,
        "n_cited": len(cited),
        "is_error": is_error,
        "states_not_found": states_not_found,
        "adversarial_pass": passed,
    }