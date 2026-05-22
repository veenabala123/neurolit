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

# Load neurolit_agent/.env so the Gemini API key is available.
# `adk web` does this automatically; a plain script must do it explicitly,
# and it must happen BEFORE the ADK / agent imports below.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "neurolit_agent" / ".env")

from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from neurolit_agent.agent import root_agent
from eval.questions import EVAL_QUESTIONS
from eval.scorers import score_adversarial, score_citations_structured

APP_NAME = "neurolit_eval"
REPORT_PATH = Path("eval/reports")


async def run_one_question(
    runner: InMemoryRunner, question: dict[str, Any]
) -> dict[str, Any]:
    """Send one question to the agent and capture its result.

    Returns a dict with:
      - answer_text: the agent's final response text
      - verified_citations: the structured citation list from the
        verify_and_finalize_citations tool response, or [] if the agent
        never called it (correct for adversarial questions).

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
    verified_citations: list[dict[str, Any]] = []
    event_n = 0

    async for event in runner.run_async(
        user_id="eval-user",
        session_id=session.id,
        new_message=user_message,
    ):
        event_n += 1
        is_final = getattr(event, "is_final_response", None)
        is_final_val = is_final() if callable(is_final) else is_final

        if not (event.content and event.content.parts):
            continue

        for part in event.content.parts:
            # Final answer text.
            text = getattr(part, "text", None)
            if text and is_final_val:
                final_text = text

            # Capture the verify_and_finalize_citations tool RESPONSE.
            # This is the structured, exact citation list - no prose parsing.
            fn_response = getattr(part, "function_response", None)
            if fn_response is not None:
                name = getattr(fn_response, "name", "")
                if name == "verify_and_finalize_citations":
                    resp = getattr(fn_response, "response", {}) or {}
                    cites = resp.get("citations", [])
                    if cites:
                        verified_citations = cites

        # Compact diagnostic - one line per event.
        kinds = []
        for part in event.content.parts:
            if getattr(part, "text", None):
                kinds.append("text")
            elif getattr(part, "function_call", None):
                fc = part.function_call
                kinds.append(f"call:{getattr(fc, 'name', '?')}")
            elif getattr(part, "function_response", None):
                fr = part.function_response
                kinds.append(f"resp:{getattr(fr, 'name', '?')}")
        print(f"    event {event_n}: final={is_final_val} parts={kinds}")

    return {
        "answer_text": final_text,
        "verified_citations": verified_citations,
    }


def score_one(
    question: dict[str, Any], run_result: dict[str, Any]
) -> dict[str, Any]:
    """Apply the right scorer based on the question tier.

    `run_result` is the dict from run_one_question: answer_text plus the
    structured verified_citations from the verify tool.
    """
    answer_text = run_result["answer_text"]
    verified = run_result["verified_citations"]

    if question["tier"] == "adversarial":
        scores = score_adversarial(answer_text, verified)
    else:
        scores = score_citations_structured(
            verified, question.get("gold_pmids", [])
        )
    return {
        "id": question["id"],
        "tier": question["tier"],
        "question": question["question"],
        "answer_text": answer_text,
        "scores": scores,
    }


def _save_raw_answers(results: list[dict[str, Any]]) -> Path:
    """Dump every raw agent answer to one file, for inspecting citation format."""
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    path = REPORT_PATH / f"raw_answers_{stamp}.md"
    blocks: list[str] = []
    for r in results:
        blocks.append(f"{'=' * 70}\n## {r['id']}\n{'=' * 70}\n")
        blocks.append(r["answer_text"] or "(empty)")
        blocks.append("\n\n")
    path.write_text("\n".join(blocks))
    return path


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
            if s.get("is_error"):
                lines.append("- ⚠ Agent errored on this question "
                             "(not a genuine refusal)")
            lines.append(f"- Citations produced: {s['n_cited']} "
                         f"(should be 0)")
            lines.append(f"- States 'not found': {s['states_not_found']}")
        else:
            n_nf = len(s["not_found_pmids"])
            n_cf = len(s["check_failed_pmids"])
            lines.append(f"- Citations used: {s['n_cited']}  "
                         f"({s['n_main']} main, {s['n_related']} related)")
            lines.append(f"- Resolved: {len(s['resolved_pmids'])}, "
                         f"not found: {n_nf}, check failed: {n_cf}")
            lines.append(f"- Citation precision: "
                         f"{_fmt_pct(s['citation_precision'])}")
            lines.append(f"- Hallucination rate: "
                         f"{_fmt_pct(s['hallucination_rate'])}")
            if s["recall"] is not None:
                lines.append(f"- Recall vs gold {s['gold_pmids']}: "
                             f"{_fmt_pct(s['recall'])} "
                             f"(found {s['gold_found']})")
            if s["not_found_pmids"]:
                lines.append(f"- ⚠ PMIDs that did not resolve "
                             f"(possible hallucinations): {s['not_found_pmids']}")
            if s["check_failed_pmids"]:
                lines.append(f"- ⚠ PMIDs whose resolution check failed "
                             f"(excluded from metrics, not hallucinations): "
                             f"{s['check_failed_pmids']}")
        lines.append("")

    path.write_text("\n".join(lines))
    return path


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    results: list[dict[str, Any]] = []

    for i, question in enumerate(EVAL_QUESTIONS, start=1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] running {question['id']} ...")
        try:
            run_result = await run_one_question(runner, question)
        except Exception as exc:  # noqa: BLE001 - eval should not crash on one bad q
            print(f"  ERROR on {question['id']}: {exc}")
            run_result = {
                "answer_text": f"(eval error: {exc})",
                "verified_citations": [],
            }
        result = score_one(question, run_result)
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
    raw_path = _save_raw_answers(results)
    print(f"\nReport written to {report_path}")
    print(f"Raw answers written to {raw_path}")


if __name__ == "__main__":
    asyncio.run(main())