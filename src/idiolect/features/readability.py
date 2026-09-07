"""Readability feature extraction for idiolect.

Extracts features related to reading ease, grade levels, and complexity.
"""

import textstat

from ..models import Document


def extract_readability(doc: Document) -> dict[str, float]:
    """Extract readability metrics from a Document."""
    features = {
        "flesch_reading_ease": 0.0,
        "flesch_kincaid_grade": 0.0,
        "gunning_fog": 0.0,
        "smog_index": 0.0,
        "coleman_liau_index": 0.0,
        "automated_readability_index": 0.0,
        "dale_chall_score": 0.0,
        "avg_syllables_per_word": 0.0,
        "polysyllable_ratio": 0.0,
        "monosyllable_ratio": 0.0,
    }

    if not doc.tokens or len(doc.tokens) < 10:
        return features

    text = doc.cleaned_text

    try:
        features["flesch_reading_ease"] = float(textstat.flesch_reading_ease(text))
        features["flesch_kincaid_grade"] = float(textstat.flesch_kincaid_grade(text))
        features["gunning_fog"] = float(textstat.gunning_fog(text))
        features["smog_index"] = float(textstat.smog_index(text))
        features["coleman_liau_index"] = float(textstat.coleman_liau_index(text))
        features["automated_readability_index"] = float(textstat.automated_readability_index(text))
        features["dale_chall_score"] = float(textstat.dale_chall_readability_score(text))

        avg_syllables = textstat.avg_syllables_per_word(text)
        features["avg_syllables_per_word"] = float(avg_syllables)

        # Polysyllable and monosyllable counts
        poly_count = textstat.polysyllabcount(text)
        mono_count = textstat.monosyllabcount(text)
        total_words = textstat.lexicon_count(text, removepunct=True)

        if total_words > 0:
            features["polysyllable_ratio"] = float(poly_count / total_words)
            features["monosyllable_ratio"] = float(mono_count / total_words)

    except Exception:
        pass

    return features
