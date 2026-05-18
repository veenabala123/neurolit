"""Retry logic for Gemini API calls.

Day 1 evals showed we hit the free-tier rate limit (5 req/min) easily.
Day 2 adds *more* LLM calls per user question (grounding + relevance),
so retry-on-429 is now load-bearing.

Strategy: parse the suggested retry delay from the Google API error,
sleep that long (with jitter), retry up to MAX_ATTEMPTS times.
Other errors propagate immediately - we don't want to mask real bugs.
"""

from __future__ import annotations

import random
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 20.0  # If we can't parse the suggested delay.
MAX_BACKOFF_SECONDS = 60.0


def _extract_retry_delay(error_message: str) -> float:
    """Pull the suggested retry delay out of a Google 429 error message.

    The Google API returns something like:
        "Please retry in 15.568891596s."
    or in the structured details:
        "retryDelay": "15s"
    """
    # First try the human-readable form.
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_message)
    if match:
        return float(match.group(1))

    # Fall back to the structured form.
    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s?", error_message)
    if match:
        return float(match.group(1))

    return DEFAULT_BACKOFF_SECONDS


def _is_retryable_error(exc: BaseException) -> bool:
    """Identify transient errors worth retrying.

    Two classes:
      - 429 RESOURCE_EXHAUSTED: rate limit hit. Retryable on per-minute limits
        (the suggested delay tells us how long). Per-day quota is also a 429
        but no amount of retrying within the day will help; we still retry
        because we can't tell the two apart from the error alone, and the
        retry will fail fast.
      - 503 UNAVAILABLE: Google's servers are overloaded. Usually resolves
        in seconds. Worth retrying.

    Non-transient errors (400 bad request, 401 auth, 404 not found, etc.)
    propagate immediately so we don't mask real bugs.
    """
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "rate limit" in text
        or "503" in text
        or "unavailable" in text
    )


def with_retry(fn: Callable[[], T], *, label: str = "llm_call") -> T:
    """Run `fn()` with retry on rate-limit errors.

    Args:
        fn: A zero-arg callable wrapping the API call.
        label: Label for log lines; helps when grepping which call retried.

    Returns:
        Whatever `fn` returns on success.

    Raises:
        The last exception if all retries are exhausted, or any non-429 error
        immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                break
            delay = min(_extract_retry_delay(str(exc)), MAX_BACKOFF_SECONDS)
            delay += random.uniform(0, 2)  # Jitter so parallel agents don't sync up.
            print(f"[retry] {label} retryable error on attempt {attempt}; "
                  f"sleeping {delay:.1f}s")
            time.sleep(delay)

    assert last_exc is not None  # Loop only exits via return or this point.
    raise last_exc