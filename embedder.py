"""
Embedder module.

Uses TF-IDF sparse embeddings - works fully offline, no model downloads.
In a real project you'd swap this for:
  - sentence-transformers (local, free, much better quality)
  - Voyage AI (Anthropic's recommended embedding provider)
  - OpenAI text-embedding-3-small

TF-IDF is good enough to demonstrate the eval framework and shows
you understand the concept. The architecture is the same either way.
"""

import numpy as np
import math
import re
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "of", "to",
    "and", "or", "for", "with", "this", "that", "are", "was",
    "be", "by", "as", "at", "from", "have", "has", "do", "not"
}


def _tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    return [t for t in tokens if t not in STOPWORDS]


class TFIDFEmbedder:
    """
    Simple TF-IDF embedder. Vocab grows as you embed more text.
    Works offline, no dependencies beyond numpy.
    """
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self._corpus = []

    def _update(self, texts):
        for text in texts:
            for token in _tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        self._corpus.extend(texts)
        N = len(self._corpus)
        df = Counter()
        for doc in self._corpus:
            for tok in set(_tokenize(doc)):
                df[tok] += 1
        self.idf = {
            tok: math.log((N + 1) / (df[tok] + 1)) + 1
            for tok in self.vocab
        }

    def _vectorize(self, text):
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = np.zeros(len(self.vocab))
        for tok, cnt in tf.items():
            if tok in self.vocab:
                vec[self.vocab[tok]] = (cnt / total) * self.idf.get(tok, 1.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, text):
        self._update([text])
        return self._vectorize(text)

    def encode_batch(self, texts):
        self._update(texts)
        return [self._vectorize(t) for t in texts]


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TFIDFEmbedder()
    return _embedder


def embed(text):
    return get_embedder().encode(text)


def embed_batch(texts):
    return get_embedder().encode_batch(texts)


def cosine_similarity(a, b):
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = np.pad(a, (0, max_len - len(a)))
        b = np.pad(b, (0, max_len - len(b)))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
