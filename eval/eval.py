"""End-to-end evaluation harness.

Runs the curated questions in ``questions.json`` against the live
``/ask`` endpoint and prints a pass/fail summary plus per-test detail.

Usage:
    # Make sure the API is running first.
    python -m eval.eval
    python -m eval.eval --base-url http://localhost:8000
    python -m eval.eval --questions eval/questions.json --json out.json

Three coarse checks run on each test:

    1. Intent match — was the agent routed correctly?
    2. Source match — does at least one expected document appear in sources?
    3. Content match — does the answer text contain at least one expected
       keyword (or the refusal sentence, when expected)?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_QUESTIONS = Path(__file__).resolve().parent / "questions.json"
DEFAULT_BASE_URL = "http://localhost:8000"

REFUSAL_SENTINEL = "could not find this information in the provided documents"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    id: str
    question: str
    answer: str
    intent: str
    confidence: str
    sources: list[dict]
    checks: list[Check] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# Checks


def _check_intent(payload: dict, expected: str | None) -> Check:
    if not expected:
        return Check("intent", True, "(not asserted)")
    actual = payload.get("intent")
    return Check(
        "intent",
        actual == expected,
        f"expected={expected} actual={actual}",
    )


def _check_sources(payload: dict, expected_docs: list[str]) -> Check:
    if not expected_docs:
        return Check("sources", True, "(not asserted)")
    sources = payload.get("sources") or []
    actual_docs = {s.get("document") for s in sources}
    overlap = [d for d in expected_docs if d in actual_docs]
    return Check(
        "sources",
        bool(overlap),
        f"expected_any={expected_docs} actual={sorted(actual_docs)}",
    )


def _check_content(payload: dict, q: dict) -> Check:
    answer = (payload.get("answer") or "").lower()

    if q.get("must_be_out_of_scope"):
        ok = payload.get("intent") == "out_of_scope"
        return Check("content", ok, "expected out_of_scope refusal")

    if q.get("must_be_refusal_or_safety"):
        keywords = [k.lower() for k in (q.get("must_include_any") or [])]
        ok = REFUSAL_SENTINEL in answer or any(k in answer for k in keywords)
        return Check("content", ok, f"refusal/safety, keywords={keywords}")

    keywords = [k.lower() for k in (q.get("must_include_any") or [])]
    if not keywords:
        return Check("content", True, "(no keyword assertions)")
    ok = any(k in answer for k in keywords)
    return Check("content", ok, f"any_of={keywords}")


# ---------------------------------------------------------------------------
# Runner


def run_one(client: httpx.Client, base_url: str, q: dict) -> TestResult:
    started = time.perf_counter()
    response = client.post(f"{base_url}/ask", json={"question": q["question"]})
    response.raise_for_status()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    payload = response.json()

    result = TestResult(
        id=q["id"],
        question=q["question"],
        answer=payload.get("answer", ""),
        intent=payload.get("intent", ""),
        confidence=payload.get("confidence", ""),
        sources=payload.get("sources") or [],
        elapsed_ms=elapsed_ms,
    )
    result.checks = [
        _check_intent(payload, q.get("expected_intent")),
        _check_sources(payload, q.get("expected_documents") or []),
        _check_content(payload, q),
    ]
    return result


def _print_result(r: TestResult) -> None:
    badge = "PASS" if r.passed else "FAIL"
    print(f"[{badge}] {r.id} ({r.elapsed_ms} ms) — intent={r.intent} confidence={r.confidence}")
    print(f"    Q: {r.question}")
    print(f"    A: {r.answer[:160].replace(chr(10), ' ')}{'…' if len(r.answer) > 160 else ''}")
    for c in r.checks:
        prefix = "  ok" if c.passed else "  X "
        print(f"    {prefix} {c.name}: {c.detail}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the /ask endpoint.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--json", help="Optional path to write structured results.")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Questions file not found: {questions_path}", file=sys.stderr)
        return 2

    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    if not questions:
        print("No questions to run.", file=sys.stderr)
        return 2

    print(f"Running {len(questions)} evals against {args.base_url}\n")

    results: list[TestResult] = []
    with httpx.Client(timeout=120) as client:
        # Quick health check up front so failures are explicit.
        try:
            client.get(f"{args.base_url}/health").raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"Health check failed against {args.base_url}: {exc}", file=sys.stderr)
            return 2

        for q in questions:
            try:
                result = run_one(client, args.base_url, q)
            except httpx.HTTPError as exc:
                result = TestResult(
                    id=q["id"],
                    question=q["question"],
                    answer=f"REQUEST FAILED: {exc}",
                    intent="",
                    confidence="",
                    sources=[],
                    checks=[Check("request", False, str(exc))],
                )
            results.append(result)
            _print_result(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"=== Summary: {passed}/{total} passed ({passed / total * 100:.1f}%) ===")

    if args.json:
        out = [
            {
                "id": r.id,
                "question": r.question,
                "answer": r.answer,
                "intent": r.intent,
                "confidence": r.confidence,
                "sources": r.sources,
                "elapsed_ms": r.elapsed_ms,
                "passed": r.passed,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in r.checks
                ],
            }
            for r in results
        ]
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote structured results to {args.json}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
