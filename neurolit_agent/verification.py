"""Citation verification: description grounding and relevance scoring.

These are the Day 2 fixes for the failure patterns Day 1 manual eval surfaced:

- Description rewriting (Novelli case): the agent shapes a paper's description
  to fit the user's question, even when the paper does something different.
  Fix: grounding check that compares description against abstract.

- Relevance stretch (Guo case): the agent includes a tangentially related paper
  in the main citation list.
  Fix: relevance score 1-5 on directness of fit to the question.

Both checks are LLM calls with tightly structured prompts and Pydantic-validated
output. They use a DIFFERENT model from the main agent on purpose - it's a weak
form of LLM-as-judge with reduced self-evaluation bias. (Same family for now;
truly different vendors come later when we have the eval harness to compare.)
"""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types as genai_types

from .retry import with_retry
from .schemas import GroundingResult, RelevanceResult

# A different Gemini model from the main agent for a touch of independence.
# Both are still Gemini, so this is not a strong cross-vendor judge. But it
# at least uses a separate decoding process and different model card.
VERIFIER_MODEL = "gemini-flash-lite-latest"

# Module-level client; created lazily because tests/imports shouldn't hit the API.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _structured_generate(
    *,
    prompt: str,
    response_schema: type,
    label: str,
) -> Any:
    """Call Gemini with a JSON-schema-constrained response.

    Returns the parsed Pydantic model instance.
    """
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        # Low temperature for verification - we want consistency, not creativity.
        temperature=0.1,
    )

    def call():
        return _get_client().models.generate_content(
            model=VERIFIER_MODEL,
            contents=prompt,
            config=config,
        )

    response = with_retry(call, label=label)

    # google-genai returns parsed Pydantic objects when response_schema is set.
    if hasattr(response, "parsed") and response.parsed is not None:
        return response.parsed

    # Fallback: parse the JSON text ourselves.
    return response_schema.model_validate(json.loads(response.text))


GROUNDING_PROMPT_TEMPLATE = """You are checking whether a one-line description of a research paper is supported by the paper's abstract.

PAPER ABSTRACT:
{abstract}

PROPOSED DESCRIPTION:
"{description}"

Is the description supported by the abstract? A description is supported if its specific factual claims appear in, or are directly entailed by, the abstract. A description that overstates what the paper does, attributes claims the abstract doesn't make, or mischaracterizes the paper's main contribution is NOT supported.

If the description is not supported, write a faithful rewrite that accurately reflects what the abstract actually says, in one sentence.

Be strict. The cost of falsely passing a bad description is higher than the cost of flagging a borderline one.
"""


def verify_description(*, abstract: str, description: str) -> GroundingResult:
    """Check whether a citation's description is supported by its abstract."""
    prompt = GROUNDING_PROMPT_TEMPLATE.format(
        abstract=abstract.strip(),
        description=description.strip(),
    )
    return _structured_generate(
        prompt=prompt,
        response_schema=GroundingResult,
        label="verify_description",
    )


RELEVANCE_PROMPT_TEMPLATE = """You are scoring how directly a research paper answers a user's specific question.

USER'S QUESTION:
"{question}"

PAPER:
Title: {title}
Authors: {authors}
Year: {year}
Journal: {journal}

ABSTRACT:
{abstract}

Score the paper's directness of fit, 1 to 5:

5 - Direct answer. The paper IS the answer (e.g. it is the foundational paper for a discovery question, or its main contribution explicitly addresses the question).
4 - Strong fit. The paper substantially addresses the question, even if not its main focus.
3 - Partial fit. The paper touches on the question but is mainly about something else.
2 - Adjacent. The paper is in the same area but doesn't really answer the question.
1 - Not relevant. The paper only mentions the topic in passing or not at all.

IMPORTANT - read carefully:

For questions about WHO, WHEN, or WHERE something was first discovered or described, the PRIMARY paper (the one that first reported the finding) is a 5, even if the abstract doesn't restate that it was the first. Use the title, authors, year, and journal to judge whether this is the primary source. Foundational papers rarely announce themselves as foundational in their own abstracts.

For methodological-comparison questions (e.g. "which papers compare X and Y"), a paper that USES one method without COMPARING it to alternatives is a 3 - not a 4.

A 2023 review claiming "X was discovered in 2005" is at most a 4 for a "when was X discovered" question - the primary 2005 paper is the 5.
"""


def score_relevance(
    *,
    question: str,
    abstract: str,
    title: str = "(not provided)",
    authors: str = "(not provided)",
    year: str = "(not provided)",
    journal: str = "(not provided)",
) -> RelevanceResult:
    """Score how directly a paper answers the user's question (1-5).

    Metadata (title, authors, year, journal) is part of the relevance signal:
    a 2005 paper by the Mosers in Nature IS the answer to "who discovered grid
    cells in 2005" even if its abstract doesn't restate this. The earlier
    abstract-only version of this prompt failed exactly that case.
    """
    prompt = RELEVANCE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        title=title.strip(),
        authors=authors.strip(),
        year=year.strip(),
        journal=journal.strip(),
        abstract=abstract.strip(),
    )
    return _structured_generate(
        prompt=prompt,
        response_schema=RelevanceResult,
        label="score_relevance",
    )