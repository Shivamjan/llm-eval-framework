"""
Hallucination Detection

Pretty simple: if faithfulness is below threshold, flag it.
Could be fancier (NLI models, etc.) but this is honest about what it is.

Also adds a secondary check based on relevance — if the answer
is very relevant but has low faithfulness, that's a stronger hallucination signal
(model is confidently answering but not from the docs).
"""

FAITHFULNESS_THRESHOLD = 0.5
RELEVANCE_THRESHOLD = 0.4


def detect_hallucination(
    faithfulness_score: float,
    relevance_score: float,faithfulness_reason: str = "") -> tuple[bool, str]:
    """
    Returns (is_hallucinating, reason_string).

    Logic:
    - Primary: faithfulness below threshold → hallucination
    - Bonus signal: high relevance + low faithfulness = confident hallucination
      (the model sounds on-topic but is making stuff up)

    """
    if faithfulness_score < FAITHFULNESS_THRESHOLD:
        if relevance_score > 0.7 and faithfulness_score < 0.35:
            reason = (
                f"high-confidence hallucination detected: "
                f"response is on-topic (relevance={relevance_score:.2f}) "
                f"but not grounded in docs (faithfulness={faithfulness_score:.2f}). "
                f"Judge: {faithfulness_reason}"
            )
        else:
            reason = (
                f"faithfulness below threshold "
                f"({faithfulness_score:.2f} < {FAITHFULNESS_THRESHOLD}). "
                f"Judge: {faithfulness_reason}"
            )
        return True, reason

    return False, ""
