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


def pmid_resolves(pmid: str) -> str:
    """Check whether a PMID resolves to a real PubMed record.

    Returns one of three states - this matters because a transient API
    failure must NOT be scored the same as a fake PMID:
      - "resolved":   the PMID is a real PubMed record
      - "not_found":  PubMed responded, and there is no such record
      - "check_failed": the lookup itself failed (timeout, rate limit,
                        network error) - resolution is UNKNOWN

    Retries a few times before giving up with "check_failed".
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{NCBI_BASE}/esummary.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            entry = data.get("result", {}).get(pmid)
            if not entry:
                return "not_found"
            # A valid record has no 'error' key; a bad uid has one.
            return "not_found" if "error" in entry else "resolved"
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))  # back off, then retry
    # All retries exhausted - we genuinely could not check.
    print(f"    [scorer] PMID {pmid} resolution check failed: {last_error}")
    return "check_failed"


def score_citations(answer_text: str, gold_pmids: list[str]) -> dict[str, Any]:
    """Score the citation-level deterministic metrics for one answer.

    Returns a dict with cited PMIDs, which resolved, precision, hallucination
    rate, and lenient recall against the gold set. PMIDs whose resolution
    check failed are reported separately and excluded from precision and
    hallucination - they are unknown, not hallucinations.
    """
    cited = extract_pmids(answer_text)

    resolved: list[str] = []
    not_found: list[str] = []
    check_failed: list[str] = []
    for pmid in cited:
        status = pmid_resolves(pmid)
        if status == "resolved":
            resolved.append(pmid)
        elif status == "not_found":
            not_found.append(pmid)
        else:
            check_failed.append(pmid)
        time.sleep(0.34)  # Stay under NCBI's ~3 req/sec unauthenticated limit.

    # Precision and hallucination are computed only over PMIDs we could
    # actually check. A failed check is "unknown", not a hallucination.
    n_checkable = len(resolved) + len(not_found)
    precision = len(resolved) / n_checkable if n_checkable else None
    hallucination_rate = len(not_found) / n_checkable if n_checkable else None

    if gold_pmids:
        found_gold = [p for p in gold_pmids if p in cited]
        recall = len(found_gold) / len(gold_pmids)
    else:
        found_gold = []
        recall = None

    return {
        "cited_pmids": cited,
        "n_cited": len(cited),
        "resolved_pmids": resolved,
        "not_found_pmids": not_found,
        "check_failed_pmids": check_failed,
        "citation_precision": precision,
        "hallucination_rate": hallucination_rate,
        "gold_pmids": gold_pmids,
        "gold_found": found_gold,
        "recall": recall,
    }


def score_citations_structured(
    verified_citations: list[dict[str, Any]],
    gold_pmids: list[str],
) -> dict[str, Any]:
    """Score citation metrics from the structured verify_and_finalize_citations
    output, rather than by parsing PMIDs out of prose.

    `verified_citations` is the list returned by the agent's
    verify_and_finalize_citations tool: each item has pmid, description,
    relevance_score, placement. We score only citations the agent actually
    used in its answer - placement 'main' or 'related', not 'drop'.

    This avoids the entire class of regex/format-dependence bugs: the PMIDs
    are exact and structured, straight from the tool call.
    """
    used = [
        c for c in verified_citations
        if c.get("placement") in ("main", "related")
    ]
    cited = [c["pmid"] for c in used if c.get("pmid")]

    resolved: list[str] = []
    not_found: list[str] = []
    check_failed: list[str] = []
    for pmid in cited:
        status = pmid_resolves(pmid)
        if status == "resolved":
            resolved.append(pmid)
        elif status == "not_found":
            not_found.append(pmid)
        else:
            check_failed.append(pmid)
        time.sleep(0.34)  # NCBI rate limit.

    n_checkable = len(resolved) + len(not_found)
    precision = len(resolved) / n_checkable if n_checkable else None
    hallucination_rate = len(not_found) / n_checkable if n_checkable else None

    if gold_pmids:
        found_gold = [p for p in gold_pmids if p in cited]
        recall = len(found_gold) / len(gold_pmids)
    else:
        found_gold = []
        recall = None

    n_main = sum(1 for c in used if c.get("placement") == "main")
    n_related = sum(1 for c in used if c.get("placement") == "related")

    return {
        "cited_pmids": cited,
        "n_cited": len(cited),
        "n_main": n_main,
        "n_related": n_related,
        "resolved_pmids": resolved,
        "not_found_pmids": not_found,
        "check_failed_pmids": check_failed,
        "citation_precision": precision,
        "hallucination_rate": hallucination_rate,
        "gold_pmids": gold_pmids,
        "gold_found": found_gold,
        "recall": recall,
    }


def score_adversarial(
    answer_text: str,
    verified_citations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score an adversarial (no-paper-exists) question.

    Hard signal: the agent should NOT have produced any used citations
    (placement main/related) - ideally it never called the verify tool at all.
    Soft signal: the answer text explicitly states the paper was not found.
    A crashed run is detected and is not scored as a pass.
    """
    text = (answer_text or "")
    is_error = text.strip().startswith("(eval error:")

    used = [
        c for c in verified_citations
        if c.get("placement") in ("main", "related")
    ]

    lower = text.lower()
    not_found_phrases = (
        "no such paper", "could not be found", "not found", "does not exist",
        "no paper", "unable to find", "no 2024 paper", "no 2023 paper",
    )
    states_not_found = any(phrase in lower for phrase in not_found_phrases)

    # Pass = a real (non-error) response that used zero citations.
    passed = (not is_error) and len(used) == 0

    return {
        "n_cited": len(used),
        "is_error": is_error,
        "states_not_found": states_not_found,
        "adversarial_pass": passed,
    }