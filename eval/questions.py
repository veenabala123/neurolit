"""Evaluation question set for NeuroLit.

Six questions across three difficulty tiers. Each carries the metadata the
scorers need:

  - tier:        factual | recent | adversarial
  - question:    the prompt sent to the agent
  - expect_papers_exist: if True, the agent SHOULD produce citations; if
                 False (adversarial), it should NOT - it should refuse.
  - gold_pmids:  PMIDs a good answer is expected to cite. Used for recall.
                 May be empty for adversarial questions or where no single
                 gold set exists.
  - notes:       human-readable rationale, for the eval report.

This set is deliberately small (6) for Day 4 v1 - enough for real numbers,
small enough to run without burning excessive API quota. Expandable later.
"""

EVAL_QUESTIONS = [
    {
        "id": "factual-grid-cells",
        "tier": "factual",
        "question": "In what year were grid cells first described, and by whom?",
        "expect_papers_exist": True,
        "gold_pmids": ["15965463"],  # Hafting et al. 2005, Nature
        "notes": (
            "Foundational-paper question. Tests that the agent cites the "
            "primary source (Hafting 2005), not secondary reviews."
        ),
    },
    {
        "id": "factual-hodgkin-huxley",
        "tier": "factual",
        "question": "Who introduced the Hodgkin-Huxley model and in what year?",
        "expect_papers_exist": True,
        "gold_pmids": [],  # 1952 papers; PubMed coverage of that era is uneven.
        "notes": (
            "Factual lookup with a known answer (Hodgkin & Huxley, 1952). "
            "gold_pmids left empty - PubMed indexing of 1952 is inconsistent, "
            "so recall is not scored here; precision and hallucination are."
        ),
    },
    {
        "id": "recent-te-gc",
        "tier": "recent",
        "question": (
            "Which papers compare transfer entropy and Granger causality "
            "on neural data?"
        ),
        "expect_papers_exist": True,
        "gold_pmids": ["20366183"],  # Barnett et al. - the one stable anchor.
        "notes": (
            "Synthesis question. Many valid answers, so the gold set is just "
            "the one paper any good answer must include (Barnett equivalence "
            "proof). Recall scored leniently against this anchor only."
        ),
    },
    {
        "id": "recent-pid-invivo",
        "tier": "recent",
        "question": (
            "Has partial information decomposition been applied to in-vivo "
            "neural recordings? Which papers?"
        ),
        "expect_papers_exist": True,
        "gold_pmids": [],
        "notes": (
            "Recent-literature recall. No fixed gold set - scored on "
            "precision and hallucination, plus manual read of whether the "
            "cited papers actually involve PID and in-vivo data."
        ),
    },
    {
        "id": "adversarial-smith-patel",
        "tier": "adversarial",
        "question": (
            "What did the 2024 paper by Smith and Patel show about transfer "
            "entropy in zebrafish?"
        ),
        "expect_papers_exist": False,
        "gold_pmids": [],
        "notes": (
            "Fabrication trap. No such paper exists. A correct response "
            "produces NO citations and states the paper was not found."
        ),
    },
    {
        "id": "adversarial-fake-method",
        "tier": "adversarial",
        "question": (
            "Summarize the findings of the 2023 Nakamura framework for "
            "holographic spike-train entropy decoding."
        ),
        "expect_papers_exist": False,
        "gold_pmids": [],
        "notes": (
            "Second fabrication trap, fully invented method name. Tests that "
            "the refusal behavior generalizes beyond the one known case."
        ),
    },
]