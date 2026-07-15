"""Shared scoring helpers: OpenAI's English normalizer + corpus WER.

We deliberately use OpenAI's own EnglishTextNormalizer (from the openai-whisper
package) because OpenAI published Whisper's LibriSpeech WER with it, which makes
our Whisper numbers directly comparable to theirs. Corpus WER = total word edits
across all utterances / total reference words (not the mean of per-utterance WERs),
matching the standard LibriSpeech convention.
"""
from whisper.normalizers import EnglishTextNormalizer

_norm = EnglishTextNormalizer()


def tokens(text: str) -> list[str]:
    return _norm(text or "").split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def corpus_wer(pairs) -> tuple[float, int, int]:
    """pairs: iterable of (reference_text, hypothesis_text).

    Returns (wer_percent, total_errors, total_reference_words).
    An empty hypothesis scores as all-deletions (100% WER for that utterance),
    matching the benchmark's "failures counted, not hidden" rule.
    """
    errors = ref_words = 0
    for ref, hyp in pairs:
        r, h = tokens(ref), tokens(hyp)
        errors += edit_distance(r, h)
        ref_words += len(r)
    return (100.0 * errors / ref_words, errors, ref_words)
