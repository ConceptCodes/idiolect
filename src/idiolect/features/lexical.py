"""Lexical feature extraction for idiolect.

Extracts features related to vocabulary richness, diversity,
word frequencies, and hapax ratios.
"""

from collections import Counter

import numpy as np
from lexicalrichness import LexicalRichness
from wordfreq import zipf_frequency

from ..models import Document


def extract_lexical(doc: Document) -> dict[str, float]:
    """Extract lexical features from a Document."""
    features = {
        "mattr": 0.0,
        "mtld": 0.0,
        "hdd": 0.0,
        "yule_k": 0.0,
        "hapax_legomena_ratio": 0.0,
        "hapax_dislegomena_ratio": 0.0,
        "avg_word_length": 0.0,
        "word_length_std": 0.0,
        "rare_word_ratio": 0.0,
        "common_word_ratio": 0.0,
        "avg_zipf_frequency": 0.0,
    }

    if not doc.tokens or len(doc.tokens) < 10:
        return features

    n_tokens = len(doc.tokens)

    # LexicalRichness features
    try:
        lex = LexicalRichness(doc.cleaned_text)
        features["mattr"] = float(lex.mattr(window_size=min(50, n_tokens)))
        features["mtld"] = float(lex.mtld(threshold=0.72))
        if n_tokens >= 42:  # HD-D usually requires at least 42 words
            features["hdd"] = float(lex.hdd(draws=42))
    except Exception:
        pass

    # Word frequencies (lowercased)
    words_lower = [w.lower() for w in doc.tokens]
    word_counts = Counter(words_lower)

    # Hapax features
    v1 = sum(1 for count in word_counts.values() if count == 1)
    v2 = sum(1 for count in word_counts.values() if count == 2)
    features["hapax_legomena_ratio"] = v1 / n_tokens if n_tokens > 0 else 0.0
    features["hapax_dislegomena_ratio"] = v2 / n_tokens if n_tokens > 0 else 0.0

    # Yule's K (10^4 * (sum(i^2 * V_i) - n_tokens) / n_tokens^2)
    # where V_i is the number of words appearing exactly i times
    freq_of_freqs = Counter(word_counts.values())
    sum_i2_vi = sum((i**2) * vi for i, vi in freq_of_freqs.items())
    features["yule_k"] = 10000.0 * (sum_i2_vi - n_tokens) / (n_tokens**2) if n_tokens > 0 else 0.0

    # Word lengths
    lengths = [len(w) for w in doc.tokens]
    features["avg_word_length"] = float(np.mean(lengths)) if lengths else 0.0
    features["word_length_std"] = float(np.std(lengths)) if lengths else 0.0

    # Zipf frequencies
    zipfs = [zipf_frequency(w, "en") for w in doc.tokens]
    if zipfs:
        features["avg_zipf_frequency"] = float(np.mean(zipfs))
        features["rare_word_ratio"] = sum(1 for z in zipfs if z < 3.0) / len(zipfs)
        features["common_word_ratio"] = sum(1 for z in zipfs if z > 5.0) / len(zipfs)

    return features
