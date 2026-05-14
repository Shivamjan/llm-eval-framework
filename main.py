"""
main.py — Run the full eval pipeline on sample data.

Usage:
    python main.py
    python main.py --method embedding   # skip LLM judge, use embeddings only
    python main.py --case 0             # run only first test case
"""

import sys
import os
import json
import argparse
import anthropic

sys.path.insert(0, os.path.dirname(__file__))

from evaluators.relevance import score_relevance
from evaluators.faithfulness import score_faithfulness
from evaluators.hallucination import detect_hallucination
from utils.logger import log_eval
from data.sample_data import SAMPLE_CASES


def run_eval(
    query: str,
    retrieved_docs: list[str],
    llm_response: str,
    prompt_label: str = "default",
    method: str = "llm",
    client: anthropic.Anthropic = None,
    verbose: bool = True
) -> dict:
    """
    Core eval function. Takes a query + docs + response, returns scores.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query[:80]}...")
        print(f"Label: {prompt_label}")
        print(f"Method: {method}")

    # 1. relevance
    relevance = score_relevance(query, llm_response)
    if verbose:
        print(f"  Relevance:    {relevance:.3f}")

    # 2. faithfulness
    faithfulness, faith_reason = score_faithfulness(
        query, llm_response, retrieved_docs,
        method=method,
        client=client
    )
    if verbose:
        print(f"  Faithfulness: {faithfulness:.3f}  ({faith_reason})")

    # 3. hallucination detection
    is_hallucination, hall_reason = detect_hallucination(
        faithfulness, relevance, faith_reason
    )
    if verbose:
        flag = "YES" if is_hallucination else "NO"
        print(f" Hallucination: {flag}")
        if hall_reason:
            print(f"    Reason: {hall_reason}")

    result = {
        "query": query,
        "prompt_label": prompt_label,
        "relevance_score": relevance,
        "faithfulness_score": faithfulness,
        "faithfulness_reason": faith_reason,
        "hallucination_flag": is_hallucination,
        "hallucination_reason": hall_reason,
    }

    #4log it
    log_eval(
        query=query,
        retrieved_docs=retrieved_docs,
        llm_response=llm_response,
        relevance_score=relevance,
        faithfulness_score=faithfulness,
        hallucination_flag=is_hallucination,
        hallucination_reason=hall_reason,
        prompt_label=prompt_label,
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="LLM Eval Framework")
    parser.add_argument(
        "--method",
        choices=["llm", "embedding"],
        default="llm",
        help="Faithfulness scoring method (default: llm)"
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="Run only a specific test case index (default: run all)"
    )
    args = parser.parse_args()

    client = None
    if args.method == "llm":
        try:
            client = anthropic.Anthropic()
        except Exception as e:
            print(f"[warning]  Anthropic client Initizalization failed: {e}")
            print("[warning] Falling back to embedding-based faithfulness")
            args.method = "embedding"

    cases = SAMPLE_CASES
    if args.case is not None:
        cases = [SAMPLE_CASES[args.case]]

    all_results = []

    for case in cases:
        query = case["query"]
        retrieved_docs = case["retrieved_docs"]

        for label, response in case["responses"].items():
            result = run_eval(
                query=query,
                retrieved_docs=retrieved_docs,
                llm_response=response,
                prompt_label=label,
                method=args.method,
                client=client,
                verbose=True
            )
            all_results.append(result)

    # summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = len(all_results)
    hallucinations = sum(1 for r in all_results if r["hallucination_flag"])
    avg_relevance = sum(r["relevance_score"] for r in all_results) / total
    avg_faithfulness = sum(r["faithfulness_score"] for r in all_results) / total

    print(f"Total evals run:       {total}")
    print(f"Hallucinations flagged: {hallucinations}/{total}")
    print(f"Avg relevance score:   {avg_relevance:.3f}")
    print(f"Avg faithfulness score: {avg_faithfulness:.3f}")
    print(f"\nAll results logged to logs/eval_logs.db")
    print("Run `python view_logs.py` to see the full log table")


if __name__ == "__main__":
    main()
