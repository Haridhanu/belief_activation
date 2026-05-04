"""Hydrate the local HuggingFace cache with the open FinanceBench eval set.

Run once to pre-fetch; subsequent ``load_financebench()`` calls hit the cache.

Usage:
    uv run python scripts/fetch_financebench.py
"""

from __future__ import annotations

from multi_agent.utils.financebench import load_financebench


def main() -> None:
    questions = load_financebench()
    n_with_evidence = sum(1 for q in questions if q.evidence_pages)
    n_pages = sum(len(q.evidence_pages) for q in questions)
    print(
        f"FinanceBench: {len(questions)} questions, "
        f"{n_with_evidence} with evidence, {n_pages} evidence pages total"
    )
    print("Cached under ~/.cache/huggingface/ (managed by `datasets`).")


if __name__ == "__main__":
    main()
