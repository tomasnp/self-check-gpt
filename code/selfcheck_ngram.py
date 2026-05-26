"""SelfCheck-n-gram hallucination detector (Manakul et al., 2023).

Idea: if a generated sentence is factual, the same facts should reappear
across stochastically resampled generations on the same prompt. We fit a
small n-gram LM on the N samples (plus optionally the response itself) and
score each sentence by its negative log-likelihood under that LM. A high
score means the tokens of the sentence are rare across the samples, which
is taken as evidence of hallucination.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


_WORD_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


@dataclass
class NgramLM:
    """Add-k smoothed n-gram language model with a backoff to unigram counts."""

    n: int
    k: float  # Laplace / add-k smoothing
    context_counts: dict[tuple[str, ...], Counter]
    vocab: set[str]

    def logprob_token(self, token: str, context: tuple[str, ...]) -> float:
        ctx = context[-(self.n - 1):] if self.n > 1 else tuple()
        counts = self.context_counts.get(ctx)
        V = max(len(self.vocab), 1)
        if counts is None:
            # Unseen context -> fall back to uniform with smoothing.
            return math.log(self.k / (self.k * V))  # = log(1/V)
        total = sum(counts.values())
        num = counts.get(token, 0) + self.k
        den = total + self.k * V
        return math.log(num / den)


def fit_ngram(corpus: list[str], n: int = 1, k: float = 1.0) -> NgramLM:
    context_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    vocab: set[str] = set()
    for text in corpus:
        toks = tokenize(text)
        if not toks:
            continue
        padded = ["<s>"] * (n - 1) + toks + ["</s>"]
        vocab.update(toks)
        for i in range(n - 1, len(padded)):
            ctx = tuple(padded[i - n + 1 : i])
            tok = padded[i]
            context_counts[ctx][tok] += 1
    vocab.update({"<s>", "</s>"})
    return NgramLM(n=n, k=k, context_counts=dict(context_counts), vocab=vocab)


def score_sentence(
    sentence: str, lm: NgramLM, aggregator: str = "avg"
) -> float:
    """Return a hallucination score in [0, +inf): higher = more hallucinated."""
    toks = tokenize(sentence)
    if not toks:
        return 0.0
    padded = ["<s>"] * (lm.n - 1) + toks + ["</s>"]
    neg_logps: list[float] = []
    for i in range(lm.n - 1, len(padded)):
        ctx = tuple(padded[i - lm.n + 1 : i])
        tok = padded[i]
        neg_logps.append(-lm.logprob_token(tok, ctx))
    if aggregator == "avg":
        return sum(neg_logps) / len(neg_logps)
    if aggregator == "max":
        return max(neg_logps)
    raise ValueError(f"unknown aggregator: {aggregator}")


def selfcheck_ngram(
    response_sentences: list[str],
    samples: list[str],
    n: int = 1,
    k: float = 1.0,
    aggregator: str = "avg",
    include_response: bool = True,
) -> list[float]:
    """Compute one hallucination score per sentence of the response.

    The LM is fit on the N stochastic samples; we optionally also include
    the response itself in the training corpus (this matches the paper's
    setup and smooths the score for rare proper nouns that appear in the
    response).
    """
    corpus = list(samples)
    if include_response:
        corpus = corpus + [" ".join(response_sentences)]
    lm = fit_ngram(corpus, n=n, k=k)
    return [score_sentence(s, lm, aggregator=aggregator) for s in response_sentences]


def passage_score(sentence_scores: list[float], aggregator: str = "avg") -> float:
    if not sentence_scores:
        return 0.0
    if aggregator == "avg":
        return sum(sentence_scores) / len(sentence_scores)
    if aggregator == "max":
        return max(sentence_scores)
    raise ValueError(f"unknown aggregator: {aggregator}")
