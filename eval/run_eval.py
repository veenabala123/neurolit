"""NeuroLit evaluation runner.

Day 4 part 1. Drives the agent over the eval question set WITHOUT the
`adk web` UI, using ADK's Runner API directly, then scores each answer with
the deterministic scorers and writes a markdown report.

Run from the repo root:
    python -m eval.run_eval

Each question is a fresh session, so the paper cache does not leak between
questions. Expect this to take several minutes - each question is multiple
LLM calls, and the scorer rate-limits its PubMed checks.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from neurolit_agent.agent import root_agent
from eval.questions import EVAL_QUESTIONS
from eval.scorers import score_adversarial, score_citations

APP_NAME = "neurolit_eval"
REPORT_PATH = Path("eval/reports")


async def run_one_question(runner: InMemoryRunner, question: dict[str, Any]) -> str:
    """Send one question to the agent and return its final answer text.

    A fresh session per question keeps the paper cache and search counter
    isolated - question N must not benefit from question N-1's state.
    """
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id="eval-user",
    )
    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=question["question"])],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id="eval-user",
        session_id=session.id,
        new_message=user_message,
    ):
        # The agent's final answer is the last event with text content
        # authored by the agent (not a tool result).
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text
    return final_text


def score_one(question: dict[str, Any], answer_text: str) -> dict[str, Any]:
    """Apply the right scorer based on the question tier."""
    if question["tier"] == "adversarial":
        scores = score_adversarial(answer_text)
    else:
        scores = score_citations(answer_text, question.get("gold_pmids", []))
    return {
        "id": question["id"],
        "tier": question["tier"],
        "question": question["question"],
        "answer_text": answer_text,
        "scores": scores,
    }


def _fmt_pct(value: float | None) -> str:
    """Format a 0-1 ratio as a percentage, or '-' if not applicable."""
    return f"{value * 100:.0f}%" if value is not None else "-"


def write_report(results: list[dict[str, Any]]) -> Path:
    """Write a markdown eval report and return its path."""
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = REPORT_PATH / f"eval_{stamp}.md"

    lines: list[str] = []
    lines.append(f"# NeuroLit eval report — {stamp}")
    lines.append("")
    lines.append(f"Questions run: {len(results)}")
    lines.append("")

    # --- Aggregate numbers ---
    cited_results = [r for r in results if r["tier"] != "adversarial"]
    adversarial_results = [r for r in results if r["tier"] == "adversarial"]

    all_precisions = [
        r["scores"]["citation_precision"]
        for r in cited_results
        if r["scores"].get("citation_precision") is not None
    ]
    all_halluc = [
        r["scores"]["hallucination_rate"]
        for r in cited_results
        if r["scores"].get("hallucination_rate") is not None
    ]
    adv_passes = sum(
        1 for r in adversarial_results if r["scores"].get("adversarial_pass")
    )

    lines.append("## Aggregate")
    lines.append("")
    if all_precisions:
        avg_prec = sum(all_precisions) / len(all_precisions)
        lines.append(f"- Mean citation precision: {_fmt_pct(avg_prec)} "
                     f"(across {len(all_precisions)} cited questions)")
    if all_halluc:
        avg_h = sum(all_halluc) / len(all_halluc)
        lines.append(f"- Mean hallucination rate: {_fmt_pct(avg_h)}")
    if adversarial_results:
        lines.append(f"- Adversarial questions passed: "
                     f"{adv_passes}/{len(adversarial_results)}")
    lines.append("")

    # --- Per-question detail ---
    lines.append("## Per-question results")
    lines.append("")
    for r in results:
        s = r["scores"]
        lines.append(f"### {r['id']}  ({r['tier']})")
        lines.append("")
        lines.append(f"**Question:** {r['question']}")
        lines.append("")
        if r["tier"] == "adversarial":
            verdict = "PASS" if s["adversarial_pass"] else "FAIL"
            lines.append(f"- Verdict: **{verdict}**")
            lines.append(f"- Citations produced: {s['n_cited']} "
                         f"(should be 0)")
            lines.append(f"- States 'not found': {s['states_not_found']}")
        else:
            lines.append(f"- Citations: {s['n_cited']}  "
                         f"(resolved {len(s['resolved_pmids'])}, "
                         f"unresolved {len(s['unresolved_pmids'])})")
            lines.append(f"- Citation precision: "
                         f"{_fmt_pct(s['citation_precision'])}")
            lines.append(f"- Hallucination rate: "
                         f"{_fmt_pct(s['hallucination_rate'])}")
            if s["recall"] is not None:
                lines.append(f"- Recall vs gold {s['gold_pmids']}: "
                             f"{_fmt_pct(s['recall'])} "
                             f"(found {s['gold_found']})")
            if s["unresolved_pmids"]:
                lines.append(f"- ⚠ Unresolved PMIDs: {s['unresolved_pmids']}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    results: list[dict[str, Any]] = []

    for i, question in enumerate(EVAL_QUESTIONS, start=1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] running {question['id']} ...")
        try:
            answer = await run_one_question(runner, question)
        except Exception as exc:  # noqa: BLE001 - eval should not crash on one bad q
            print(f"  ERROR on {question['id']}: {exc}")
            answer = f"(eval error: {exc})"
        result = score_one(question, answer)
        results.append(result)

        # Brief console feedback per question.
        s = result["scores"]
        if question["tier"] == "adversarial":
            print(f"  -> adversarial {'PASS' if s['adversarial_pass'] else 'FAIL'}")
        else:
            print(f"  -> {s['n_cited']} citations, "
                  f"precision {_fmt_pct(s['citation_precision'])}, "
                  f"hallucination {_fmt_pct(s['hallucination_rate'])}")

    report_path = write_report(results)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())