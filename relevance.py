"""
Relevance Score

Does the LLM response actually answer the query?
Method: embed both, compute cosine similarity.

This is a blunt instrument but it works reasonably well.
A score of 0.8+ = good. 0.5-0.8 = meh. <0.5 = probably bad.
"""

from utils.embedder import embed, cosine_similarity


def score_relevance(query: str, llm_response: str) -> float:
    """
    Returns a float between 0 and 1.
    Higher = the response is more semantically similar to the query.
    """
    q_emb = embed(query)
    r_emb = embed(llm_response)
    score = cosine_similarity(q_emb, r_emb)

    # cosine sim can technically be negative, clip to [0, 1]
    return max(0.0, min(1.0, score))
