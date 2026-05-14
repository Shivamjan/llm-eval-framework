"""
compare_prompts.py

Run the same query through two different prompts and compare scores.
This is the "prompt A vs prompt B" experiment.

In practice you'd feed the prompts to an LLM and compare its outputs.
Here we simulate it with pre-written responses so you can run without burning API credits.
"""

import sys
import os
import anthropic

sys.path.insert(0, os.path.dirname(__file__))

from main import run_eval


# --- The test scenario ---

QUERY = "What are transformers used for in NLP?"

RETRIEVED_DOCS = [
    "Transformer models have become the dominant architecture for NLP tasks including machine translation, text summarization, question answering, and text classification.",
    "BERT is a transformer-based model pre-trained on masked language modeling. It is used for tasks like named entity recognition and sentiment analysis.",
    "GPT models are decoder-only transformers trained to predict the next token. They are used for text generation, code completion, and conversational AI."
]

# Prompt v1: generic, vague system prompt
PROMPT_V1_SYSTEM = "You are a helpful assistant. Answer the question."

# Prompt v2: specific, context-aware system prompt
PROMPT_V2_SYSTEM = """You are a technical AI assistant. Answer based strictly on the provided context. 
If the context does not support a claim, do not make it. Be precise and cite specifics."""


def get_llm_response(query: str, system_prompt: str, context_docs: list[str]) -> str:
    """Actually call the LLM with a prompt and context."""
    client = anthropic.Anthropic()
    context = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(context_docs)])

    user_message = f"""Context:
                    {context}

                    Question: {query}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return message.content[0].text.strip()


def compare_prompts():
    print("=" * 60)
    print("PROMPT COMPARISON EXPERIMENT")
    print("=" * 60)
    print(f"Query: {QUERY}")
    print(f"Docs: {len(RETRIEVED_DOCS)} retrieved")

    client = anthropic.Anthropic()

    print("\n[Getting response from Prompt v1...]")
    response_v1 = get_llm_response(QUERY, PROMPT_V1_SYSTEM, RETRIEVED_DOCS)
    print(f"Response v1:\n  {response_v1[:200]}...")

    print("\n[Getting response from Prompt v2...]")
    response_v2 = get_llm_response(QUERY, PROMPT_V2_SYSTEM, RETRIEVED_DOCS)
    print(f"Response v2:\n  {response_v2[:200]}...")

    # eval both
    print("\n[Evaluating both responses...]")

    result_v1 = run_eval(
        query=QUERY,
        retrieved_docs=RETRIEVED_DOCS,
        llm_response=response_v1,
        prompt_label="prompt_v1_generic",
        method="llm",
        client=client,
        verbose=True
    )

    result_v2 = run_eval(
        query=QUERY,
        retrieved_docs=RETRIEVED_DOCS,
        llm_response=response_v2,
        prompt_label="prompt_v2_context_aware",
        method="llm",
        client=client,
        verbose=True
    )

    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Prompt v1':>15} {'Prompt v2':>15}")
    print(f"{'-'*55}")
    print(f"{'Relevance score':<25} {result_v1['relevance_score']:>15.3f} {result_v2['relevance_score']:>15.3f}")
    print(f"{'Faithfulness score':<25} {result_v1['faithfulness_score']:>15.3f} {result_v2['faithfulness_score']:>15.3f}")

    h1 = "YES" if result_v1["hallucination_flag"] else "NO"
    h2 = "YES" if result_v2["hallucination_flag"] else "NO"
    print(f"{'Hallucination':<25} {h1:>15} {h2:>15}")

    print(f"\nConclusion:")
    if result_v2["faithfulness_score"] > result_v1["faithfulness_score"]:
        diff = result_v2["faithfulness_score"] - result_v1["faithfulness_score"]
        print(f"  Prompt v2 is more faithful by {diff:.3f} — context-aware prompting helps.")
    elif result_v1["faithfulness_score"] > result_v2["faithfulness_score"]:
        print("  Prompt v1 scored higher (unexpected — check your prompts)")
    else:
        print("  Both prompts performed similarly on this query.")

    print(f"\nBoth runs logged to logs/eval_logs.db")


if __name__ == "__main__":
    compare_prompts()
