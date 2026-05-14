"""
Faithfulness Score

Is the LLM response grounded in the retrieved documents?

Two methods here:
  1. Embedding-based: average cosine sim between response and each doc
  2. LLM-as-judge: ask Claude if the answer is supported by the context

Method 2 is better but costs API calls.
We use method 2 by default, with method 1 as fallback.

Score interpretation:
  >0.7  = well grounded
  0.5-0.7 = somewhat grounded, watch out
  <0.5  = probably hallucinating
"""

import anthropic
import numpy as np

from utils.embedder import embed, embed_batch, cosine_similarity


FAITHFULNESS_THRESHOLD = 0.5  # tune this as per use case


def score_faithfulness_embedding(llm_response: str, retrieved_docs: list[str]) -> float:
    """
    
    Computes average cosine similarity between the response and each retrieved doc.
    Fast and cheap but less precise than LLM-as-judge.
    """
    if not retrieved_docs:
        return 0.0

    response_emb = embed(llm_response)
    doc_embs = embed_batch(retrieved_docs)

    sims = [cosine_similarity(response_emb, doc_emb) for doc_emb in doc_embs]

    # take the max similarity (best-matching doc) rather than average
    # this makes more sense — if at least ONE doc supports it, then it's grounded
    return max(0.0, min(1.0, max(sims)))


def score_faithfulness_llm(
    query: str,
    llm_response: str,
    retrieved_docs: list[str],
    client: anthropic.Anthropic = None
) -> tuple[float, str]:
    """
    LLM-as-judge faithfulness scoring.
    Returns (score, explanation).

    Prompts Claude to rate how well the response is supported by the context.
    Score is 0.0 to 1.0 based on Claude's judgment.
    """
    if client is None:
        client = anthropic.Anthropic()

    context = "\n\n---\n\n".join(
        [f"Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    # keeping the prompt simple and direct
    judge_prompt = f"""You are evaluating whether an AI assistant's response is faithful to the provided context documents.

                        Query: {query}

                        Context Documents:
                        {context}

                        AI Response:
                        {llm_response}

                        Task: Rate how well the response is grounded in the context documents on a scale of 0.0 to 1.0.

                        Rules:
                        - 1.0 = every claim in the response is directly supported by the documents
                        - 0.7 = most claims are supported, minor extrapolations
                        - 0.5 = roughly half the claims are supported
                        - 0.3 = few claims are supported, significant unsupported content
                        - 0.0 = completely unsupported or contradicts the documents

                        Respond in exactly this format:
                        SCORE: <number between 0.0 and 1.0>
                        REASON: <one sentence explaining the score>"""






    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    response_text = message.content[0].text.strip()

    # parse the score out
    score = 0.5  # default if parsing fails
    reason = "could not parse judge response"

    for line in response_text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return score, reason


def score_faithfulness(
                query: str,
                llm_response: str,
                retrieved_docs: list[str],
                method: str = "llm",  # "llm" or "embedding"
                client: anthropic.Anthropic = None) -> tuple[float, str]:
    """
    Main faithfulness scoring function.
    Returns (score, explanation).
    """
    if method == "embedding":
        score = score_faithfulness_embedding(llm_response, retrieved_docs)
        return score, "embedding-based similarity score"
    else:
        try:
            return score_faithfulness_llm(query, llm_response, retrieved_docs, client)
        except Exception as e:
            # fall back to embedding if LLM call fails
            print(f"[faithfulness] LLM judge failed ({e}), falling back to embedding")
            score = score_faithfulness_embedding(llm_response, retrieved_docs)
            return score, f"embedding fallback (llm error: {e})"
