"""NeuroLit — Day 1: minimal ADK agent + PubMed search tool.

Run with:
    cd neurolit/         # the directory CONTAINING neurolit_agent/
    adk web

Then open http://localhost:8000 and select `neurolit_agent`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests
from google.adk.agents.llm_agent import Agent

# NCBI E-utilities base URL. No API key required for low-volume use,
# but we should respect their rate limit (3 req/sec without a key).
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_TIMEOUT = 15  # seconds


def search_pubmed(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search PubMed for neuroscience papers matching the query.

    Use this tool when the user asks a question about the neuroscience literature
    and you need to find relevant papers. Returns paper metadata including title,
    authors, year, journal, abstract, and PubMed URL. Always cite specific papers
    by their PMID and title when using results from this tool.

    Args:
        query: A search query. Use specific neuroscience terms. Boolean operators
            (AND, OR, NOT) and field tags like [Title], [Author], [MeSH Terms]
            are supported. Example: "transfer entropy AND effective connectivity".
        max_results: Maximum number of papers to return (default 5, max 20).
            Use fewer for focused queries, more for broad literature surveys.

    Returns:
        A dict with status, query, count, and a papers list. Each paper has:
        pmid, title, authors (list), year, journal, abstract, url.
        On failure, returns status="error" with an error_message.
    """
    if not query or not query.strip():
        return {"status": "error", "error_message": "query is empty"}

    max_results = max(1, min(max_results, 20))

    try:
        # Step 1: esearch returns PMIDs matching the query.
        esearch = requests.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=REQUEST_TIMEOUT,
        )
        esearch.raise_for_status()
        pmids = esearch.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return {
                "status": "success",
                "query": query,
                "count": 0,
                "papers": [],
                "note": "No papers found. Consider broader terms or synonyms.",
            }

        # Step 2: efetch returns full metadata + abstracts as XML.
        efetch = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
            timeout=REQUEST_TIMEOUT,
        )
        efetch.raise_for_status()
        papers = _parse_pubmed_xml(efetch.text)

        return {
            "status": "success",
            "query": query,
            "count": len(papers),
            "papers": papers,
        }

    except requests.RequestException as exc:
        return {
            "status": "error",
            "error_message": f"PubMed request failed: {exc}",
        }
    except (ET.ParseError, ValueError) as exc:
        return {
            "status": "error",
            "error_message": f"Failed to parse PubMed response: {exc}",
        }


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Pull title, authors, year, journal, and abstract out of a PubMed XML reply."""
    root = ET.fromstring(xml_text)
    papers: list[dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"

        # Abstracts can have multiple labeled sections (Background, Methods, ...).
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
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        })

    return papers


# The agent definition. ADK auto-discovers `root_agent`.
root_agent = Agent(
    model="gemini-flash-latest",
    name="neurolit",
    description=(
        "A neuroscience literature research assistant for researchers entering "
        "the field from adjacent quantitative disciplines."
    ),
    instruction=(
        "You are NeuroLit, a research assistant that helps researchers — particularly "
        "those entering neuroscience from adjacent fields like physics, CS, or "
        "engineering — find and synthesize the neuroscience literature.\n\n"
        "Your user has strong research methodology training but may not know "
        "neuroscience-specific jargon, canonical papers, or named research groups. "
        "Adapt your responses accordingly:\n"
        "  - When you use a neuroscience term-of-art, define it briefly.\n"
        "  - When citing a paper, give title, lead author, year, and one line on "
        "    why it matters (e.g. 'introduced method X', 'foundational review').\n"
        "  - If a term has multiple meanings across subfields (e.g. 'connectivity'), "
        "    surface the disambiguation up front.\n\n"
        "When a user asks a literature question, use the `search_pubmed` tool. "
        "Construct focused queries with precise terminology. If a first query "
        "returns too few or too many results, refine and search again — but cap "
        "yourself at 3 searches per user question.\n\n"
        "Citation discipline is critical:\n"
        "  - NEVER cite a paper that did not come from a tool call this session.\n"
        "  - Quote the PMID for every cited paper.\n"
        "  - If the tool returns no relevant results, say so plainly. Do not "
        "    invent citations to fill the gap.\n\n"
        "Format answers with a short orientation paragraph, then claims grouped "
        "by theme, then a 'Papers cited' list at the end with PMID + URL."
    ),
    tools=[search_pubmed],
)
