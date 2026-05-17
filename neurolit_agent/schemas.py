"""Shared data schemas for NeuroLit.

Pydantic models give us structured-output validation when the LLM
produces JSON, and self-documenting types throughout the codebase.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CitationDraft(BaseModel):
    """A citation the agent has drafted, before verification."""

    pmid: str
    description: str = Field(
        ...,
        description="One-line role the agent assigns this paper "
        "(e.g. 'foundational paper introducing method X').",
    )


class GroundingResult(BaseModel):
    """Output of the description-grounding check."""

    supported: bool = Field(
        ...,
        description="True if the description's specific claims appear in or are "
        "directly entailed by the abstract.",
    )
    confidence: Literal["high", "medium", "low"]
    rewritten_description: str | None = Field(
        None,
        description="If unsupported, a faithful rewrite based only on the abstract. "
        "None if the description is supported as-is.",
    )
    reason: str = Field(..., description="One sentence explaining the judgment.")


class RelevanceResult(BaseModel):
    """Output of the relevance-scoring check."""

    score: int = Field(..., ge=1, le=5, description="1=not relevant, 5=direct answer.")
    rationale: str = Field(..., description="One sentence explaining the score.")


class VerifiedCitation(BaseModel):
    """A citation after grounding and relevance checks."""

    pmid: str
    description: str
    relevance_score: int
    placement: Literal["main", "related", "drop"]
    audit: str = Field(
        ...,
        description="Short note explaining placement decision, for trace visibility.",
    )