"""NeuroLit — Day 3: PubMed search + verification + session paper cache.

Day 3 adds a session-scoped paper cache: every paper retrieved this
conversation is stored in ADK session state, keyed by PMID. The agent
avoids re-fetching papers it has already seen, and can resolve references
to earlier papers ("that Faes paper") via the list_cached_papers tool.

Run with:
    cd neurolit/
    adk web
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests
from google.adk.agents.llm_agent import Agent
from google.adk.tools import ToolContext

from .schemas import VerifiedCitation
from .verification import score_relevance, verify_description

# NCBI E-utilities base URL.
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_TIMEOUT = 15

# Relevance threshold: papers scoring at or below this go to "Related work"
# rather than the main "Papers cited" list. 4 = strong fit, which matches
# our Day 1 finding that Guo 2024 was a "3" mis-placed as a main citation.
MAIN_CITATION_MIN_SCORE = 4

# Hard cap on searches per question. Day 2 testing found the agent looped 7
# times on an adversarial query (a paper that doesn't exist) because the
# soft "cap yourself at 3 searches" instruction was ignored. This is the
# enforced version: the tool itself refuses past the cap.
MAX_SEARCHES_PER_QUESTION = 4

# Session-state key for the per-question search counter.
_SEARCH_COUNT_KEY = "search_count"

# Conference-proceedings records (abstract books) bundle hundreds of unrelated
# mini-abstracts under one PMID. They produce spurious AND-matches and their
# huge abstracts bloat the verification LLM calls. We drop them from results.
# A real paper abstract is at most ~600 words; these run to thousands.
_MAX_REASONABLE_ABSTRACT_WORDS = 1200
_PROCEEDINGS_TITLE_MARKERS = (
    "annual meeting",
    "annual computational",
    "abstracts from",
    "conference abstracts",
    "proceedings of",
    "meeting abstracts",
)


def _looks_like_proceedings(paper: dict[str, Any]) -> bool:
    """True if a record looks like a conference proceedings / abstract book.

    Two signals: an implausibly long abstract (real paper abstracts are short),
    or a title that matches a known proceedings pattern.
    """
    abstract = paper.get("abstract", "") or ""
    if len(abstract.split()) > _MAX_REASONABLE_ABSTRACT_WORDS:
        return True
    title = (paper.get("title", "") or "").lower()
    return any(marker in title for marker in _PROCEEDINGS_TITLE_MARKERS)


# ---------------------------------------------------------------------------
# Paper cache (Day 3)
#
# A session-scoped store of every paper retrieved this conversation, keyed by
# PMID, living in ADK session state (tool_context.state). Two payoffs:
#   - Avoids re-fetching papers already seen this session.
#   - Lets the agent resolve references like "that Faes paper" to a concrete
#     PMID via list_cached_papers.
# Same mechanism as the search counter, just storing a dict of papers.
# ---------------------------------------------------------------------------

_PAPER_CACHE_KEY = "paper_cache"


def _cache_papers(tool_context: ToolContext, papers: list[dict[str, Any]]) -> None:
    """Store papers in the session paper cache, keyed by PMID."""
    cache = tool_context.state.get(_PAPER_CACHE_KEY, {})
    for paper in papers:
        pmid = paper.get("pmid")
        if pmid:
            cache[pmid] = paper
    # Re-assign so ADK persists the mutation to session state.
    tool_context.state[_PAPER_CACHE_KEY] = cache


def _get_cached_paper(
    tool_context: ToolContext, pmid: str
) -> dict[str, Any] | None:
    """Return a cached paper by PMID, or None if not cached."""
    cache = tool_context.state.get(_PAPER_CACHE_KEY, {})
    return cache.get(pmid)


# ---------------------------------------------------------------------------
# Tool 1: search_pubmed (unchanged from Day 1)
# ---------------------------------------------------------------------------

def search_pubmed(
    query: str,
    tool_context: ToolContext,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search PubMed for neuroscience papers matching the query.

    Use when the user asks a literature question and you need to find papers.
    Returns paper metadata: pmid, title, authors, year, journal, abstract, url.
    Always cite specific PMIDs from this tool's results - never cite from memory.

    There is a hard limit on searches per question. If you hit it, stop
    searching and either conclude with the papers found so far, or tell the
    user the topic could not be found. Do not keep trying new queries.

    Args:
        query: PubMed search query. Boolean operators (AND, OR, NOT) and field
            tags ([Title], [Author], [MeSH Terms]) are supported.
        max_results: Max papers to return (default 5, capped at 20).

    Returns:
        Dict with status, query, count, and a papers list. If the per-question
        search limit is reached, returns status='limit_reached' instead.
    """
    if not query or not query.strip():
        return {"status": "error", "error_message": "query is empty"}

    # Enforce the per-question search cap using session state. The counter is
    # reset by verify_and_finalize_citations at the end of each question.
    count = tool_context.state.get(_SEARCH_COUNT_KEY, 0)
    if count >= MAX_SEARCHES_PER_QUESTION:
        return {
            "status": "limit_reached",
            "message": (
                f"Search limit ({MAX_SEARCHES_PER_QUESTION}) reached for this "
                "question. Do not search again. Conclude with the papers found "
                "so far, or tell the user the topic could not be found."
            ),
            "searches_used": count,
        }
    tool_context.state[_SEARCH_COUNT_KEY] = count + 1

    max_results = max(1, min(max_results, 20))

    try:
        esearch = requests.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params={
                "db": "pubmed", "term": query, "retmax": max_results,
                "retmode": "json", "sort": "relevance",
            },
            timeout=REQUEST_TIMEOUT,
        )
        esearch.raise_for_status()
        pmids = esearch.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return {
                "status": "success", "query": query, "count": 0, "papers": [],
                "note": "No papers found. Try broader terms or synonyms.",
            }

        efetch = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            timeout=REQUEST_TIMEOUT,
        )
        efetch.raise_for_status()
        papers = _parse_pubmed_xml(efetch.text)

        # Drop conference-proceedings records: they cause spurious AND-matches
        # and their huge abstracts bloat downstream verification LLM calls.
        filtered = [p for p in papers if not _looks_like_proceedings(p)]
        dropped = len(papers) - len(filtered)

        # Day 3: cache every usable paper so we don't re-fetch it this session.
        _cache_papers(tool_context, filtered)

        result: dict[str, Any] = {
            "status": "success",
            "query": query,
            "count": len(filtered),
            "papers": filtered,
        }
        if dropped:
            result["note"] = (
                f"{dropped} conference-proceedings record(s) were filtered out "
                "as not citable."
            )
        if not filtered:
            result["note"] = (
                "No usable papers found"
                + (f" ({dropped} proceedings record(s) filtered)" if dropped else "")
                + ". Try broader terms, or conclude the topic could not be found."
            )
        return result

    except requests.RequestException as exc:
        return {"status": "error", "error_message": f"PubMed request failed: {exc}"}
    except (ET.ParseError, ValueError) as exc:
        return {"status": "error", "error_message": f"Failed to parse response: {exc}"}


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Extract paper records from a PubMed XML response."""
    root = ET.fromstring(xml_text)
    papers: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"

        abstract_parts: list[str] = []
        for abs_el in article.findall(".//AbstractText"):
            label = abs_el.get("Label")
            text = "".join(abs_el.itertext()).strip()
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts) if abstract_parts else "(no abstract available)"

        authors: list[str] = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName")
            initials = author.findtext("Initials")
            if last and initials:
                authors.append(f"{last} {initials}")
            elif last:
                authors.append(last)

        year_el = article.find(".//PubDate/Year") or article.find(".//PubDate/MedlineDate")
        year = year_el.text[:4] if year_el is not None and year_el.text else "n.d."

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else "(unknown journal)"

        papers.append({
            "pmid": pmid, "title": title, "authors": authors, "year": year,
            "journal": journal, "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        })
    return papers


# ---------------------------------------------------------------------------
# Tool 2: fetch_abstract (Day 2 new)
# ---------------------------------------------------------------------------

def fetch_abstract(pmid: str, tool_context: ToolContext) -> dict[str, Any]:
    """Fetch a single paper's metadata + abstract by PMID.

    Use this when you need to re-check what a specific paper actually says,
    for example when verifying a citation's description. Papers already seen
    this session are served from the session cache without a network call.

    Args:
        pmid: PubMed ID, e.g. "15965463".

    Returns:
        Dict with status, and on success the paper dict (same shape as
        a single entry in search_pubmed's papers list). A 'from_cache' flag
        indicates whether the paper was served from the session cache.
    """
    pmid = (pmid or "").strip()
    if not pmid.isdigit():
        return {"status": "error", "error_message": f"invalid PMID: {pmid!r}"}

    # Day 3: serve from the session cache if we already have this paper.
    cached = _get_cached_paper(tool_context, pmid)
    if cached is not None:
        return {"status": "success", "paper": cached, "from_cache": True}

    try:
        efetch = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=REQUEST_TIMEOUT,
        )
        efetch.raise_for_status()
        papers = _parse_pubmed_xml(efetch.text)
        if not papers:
            return {"status": "error", "error_message": f"PMID {pmid} not found"}
        # Cache it so a later fetch this session is free.
        _cache_papers(tool_context, papers)
        return {"status": "success", "paper": papers[0], "from_cache": False}
    except requests.RequestException as exc:
        return {"status": "error", "error_message": f"PubMed fetch failed: {exc}"}


def list_cached_papers(tool_context: ToolContext) -> dict[str, Any]:
    """List papers already retrieved earlier in this conversation.

    Use this to resolve references to papers from earlier turns - e.g. if the
    user says "tell me more about that Faes paper" or "the second one you
    mentioned", check here for the PMID instead of searching again.

    Returns:
        Dict with status, count, and a 'papers' list of {pmid, title, authors,
        year, journal} summaries (abstracts omitted to keep the list compact).
    """
    cache = tool_context.state.get(_PAPER_CACHE_KEY, {})
    summaries = [
        {
            "pmid": p.get("pmid", ""),
            "title": p.get("title", ""),
            "authors": p.get("authors", []),
            "year": p.get("year", ""),
            "journal": p.get("journal", ""),
        }
        for p in cache.values()
    ]
    return {"status": "success", "count": len(summaries), "papers": summaries}


# ---------------------------------------------------------------------------
# Tool 3: verify_and_finalize_citations (Day 2 new - the core fix)
# ---------------------------------------------------------------------------

def verify_and_finalize_citations(
    question: str,
    draft_citations: list[dict[str, str]],
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Verify draft citations and produce the final, audited list.

    For each draft citation, this:
      1. Fetches the paper's abstract by PMID.
      2. Checks whether the proposed description is supported by the abstract.
         If not, replaces it with a faithful rewrite.
      3. Scores how directly the paper answers the user's original question.
      4. Decides placement: 'main' (score >= 4), 'related' (score 2-3),
         or 'drop' (score 1, or PMID couldn't be fetched).

    YOU (the agent) MUST call this before producing your final answer.
    Use only the returned 'main' citations in the main answer body, and the
    'related' citations in a clearly labeled 'Related work' section.

    Calling this also resets the per-question search counter, so the next
    question starts with a fresh search budget.

    Args:
        question: The user's original question, verbatim. Used for relevance scoring.
        draft_citations: List of {"pmid": "...", "description": "..."} dicts.

    Returns:
        Dict with status and a 'citations' list of verified records.
    """
    # This call marks the end of work on one question: reset the search
    # counter so the next question gets its own search budget.
    tool_context.state[_SEARCH_COUNT_KEY] = 0

    if not question or not question.strip():
        return {"status": "error", "error_message": "question is empty"}
    if not draft_citations:
        return {"status": "success", "citations": []}

    verified: list[dict[str, Any]] = []
    for draft in draft_citations:
        pmid = (draft.get("pmid") or "").strip()
        description = (draft.get("description") or "").strip()
        if not pmid or not description:
            continue

        fetch = fetch_abstract(pmid, tool_context)
        if fetch["status"] != "success":
            verified.append(VerifiedCitation(
                pmid=pmid, description=description, relevance_score=0,
                placement="drop",
                audit=f"Could not fetch PMID {pmid}: {fetch.get('error_message', 'unknown')}",
            ).model_dump())
            continue

        abstract = fetch["paper"]["abstract"]
        paper = fetch["paper"]

        # Step 1: description grounding.
        grounding = verify_description(abstract=abstract, description=description)
        final_description = description
        audit_parts: list[str] = []
        if not grounding.supported:
            if grounding.rewritten_description:
                final_description = grounding.rewritten_description
                audit_parts.append(
                    f"description rewritten ({grounding.confidence} confidence): "
                    f"{grounding.reason}"
                )
            else:
                audit_parts.append(f"description unsupported: {grounding.reason}")
        else:
            audit_parts.append(f"description supported ({grounding.confidence})")

        # Step 2: relevance scoring. Pass metadata so primary sources score
        # correctly on "who/when/where" questions where the abstract alone
        # is insufficient (a 2005 paper IS the answer to "discovered in 2005").
        relevance = score_relevance(
            question=question,
            abstract=abstract,
            title=paper.get("title", ""),
            authors=", ".join(paper.get("authors", [])),
            year=str(paper.get("year", "")),
            journal=paper.get("journal", ""),
        )
        audit_parts.append(f"relevance={relevance.score}/5: {relevance.rationale}")

        # Step 3: placement decision.
        if relevance.score >= MAIN_CITATION_MIN_SCORE:
            placement = "main"
        elif relevance.score >= 2:
            placement = "related"
        else:
            placement = "drop"

        verified.append(VerifiedCitation(
            pmid=pmid,
            description=final_description,
            relevance_score=relevance.score,
            placement=placement,
            audit=" | ".join(audit_parts),
        ).model_dump())

    return {"status": "success", "citations": verified}


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-flash-latest",
    name="neurolit",
    description=(
        "A neuroscience literature research assistant for researchers entering "
        "the field from adjacent quantitative disciplines."
    ),
    instruction=(
        "You are NeuroLit, a research assistant that helps researchers — "
        "particularly those entering neuroscience from adjacent fields like "
        "physics, CS, or engineering — find and synthesize the neuroscience "
        "literature.\n\n"

        "Your user has strong research methodology training but may not know "
        "neuroscience jargon, canonical papers, or named research groups. "
        "When you use a neuroscience term-of-art, define it briefly. If a "
        "term has multiple meanings across subfields (e.g. 'connectivity'), "
        "surface the disambiguation up front.\n\n"

        "## Workflow for every literature question\n\n"
        "1. Use `search_pubmed` to find candidate papers. Construct focused "
        "queries; refine if needed. There is a hard limit of 4 searches per "
        "question. If a search returns status='limit_reached', STOP searching "
        "immediately - conclude with the papers you have, or if none are "
        "relevant, tell the user the topic or paper could not be found.\n"
        "2. **If searches return no usable papers** (count=0, or only results "
        "that clearly don't match the question), do NOT call "
        "`verify_and_finalize_citations`. There is nothing to verify. Skip "
        "straight to telling the user the paper or topic could not be found, "
        "and if the question assumed a specific paper exists, say plainly that "
        "no such paper was found. Stop there.\n"
        "3. Otherwise, from the search results, draft a list of citations you "
        "intend to use. Each draft is a `{'pmid': '...', 'description': '...'}` "
        "where description is your one-line role for that paper.\n"
        "4. **REQUIRED (only when you have real draft citations):** call "
        "`verify_and_finalize_citations` with the user's original question and "
        "your draft citations. This runs description-grounding and relevance-"
        "scoring; you MUST do this before producing a cited final answer.\n"
        "5. Compose the final answer using ONLY the verified citation list:\n"
        "   - Papers with placement='main' go in the main 'Papers cited' "
        "section and may be referenced throughout the answer.\n"
        "   - Papers with placement='related' go in a 'Related work' section "
        "at the end, with a brief note that they touch on the topic but don't "
        "directly answer the question.\n"
        "   - Papers with placement='drop' MUST NOT appear in the output.\n"
        "   - Use the description field from the verified list, NOT your "
        "original draft - the verifier may have rewritten it for accuracy.\n\n"

        "## Citation discipline (critical)\n\n"
        "- NEVER cite a paper that did not come from a `search_pubmed` call "
        "in this session.\n"
        "- NEVER include a citation in the final answer that you did not pass "
        "through `verify_and_finalize_citations`.\n"
        "- If `verify_and_finalize_citations` returns zero main citations, "
        "tell the user the search did not surface papers that directly answer "
        "their question, list the related papers if any, and suggest how to "
        "refine the query.\n"
        "- If the user's question has a false premise (e.g. asks about a paper "
        "that doesn't exist), say so plainly. Do not invent citations.\n\n"

        "## Follow-up questions and session memory\n\n"
        "Papers retrieved earlier in the conversation are remembered. If the "
        "user refers to a paper from an earlier turn (e.g. 'that Faes paper', "
        "'the second one', 'the grid cell paper you mentioned'), call "
        "`list_cached_papers` to find its PMID instead of searching again, "
        "then use `fetch_abstract` to get its details. Only run a new "
        "`search_pubmed` if the cache does not already contain what's needed.\n\n"

        "## Output format\n\n"
        "Short orientation paragraph (1-2 sentences framing the question and "
        "any term disambiguations). Then claims grouped by theme, referencing "
        "main citations by PMID. Then 'Papers cited' list with PMID + URL "
        "for main citations. Then 'Related work' if any."
    ),
    tools=[search_pubmed, fetch_abstract, list_cached_papers,
           verify_and_finalize_citations],
)