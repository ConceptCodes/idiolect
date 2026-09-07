"""Pragmatic and discourse feature extraction for idiolect.

Extracts features related to pronoun usage, stance, formality,
discourse markers, contractions, and sentiment.
"""

import re

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..models import Document

_vader_analyzer: SentimentIntensityAnalyzer | None = None

# Matches verbal contractions like don't, they're, we've, he'll, I'd, I'm, it's, that's
# while avoiding general noun possessives like "company's" or "grandfather's"
_CONTRACTION_PATTERN = re.compile(
    r"\b(?:[a-zA-Z]+n't|[a-zA-Z]+'re|[a-zA-Z]+'ve|[a-zA-Z]+'ll|[a-zA-Z]+'d|[a-zA-Z]+'m|(?:it|that|he|she|who|what|there|here|let)'s)\b",
    re.IGNORECASE,
)


def _get_vader_analyzer() -> SentimentIntensityAnalyzer:
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def extract_pragmatic(doc: Document) -> dict[str, float]:
    """Extract pragmatic and discourse features from a Document."""
    features = {
        "pron_first_singular": 0.0,
        "pron_first_plural": 0.0,
        "pron_second": 0.0,
        "pron_third": 0.0,
        "self_immersion_index": 0.0,
        "audience_engagement_index": 0.0,
        "hedge_ratio": 0.0,
        "booster_ratio": 0.0,
        "hedge_booster_ratio": 0.0,
        "formality_score": 0.0,
        "discourse_marker_ratio": 0.0,
        "contraction_ratio": 0.0,
        "sentiment_mean": 0.0,
        "sentiment_std": 0.0,
        "sentiment_positive_ratio": 0.0,
        "sentiment_negative_ratio": 0.0,
    }

    if not doc.tokens or not doc.sentences:
        return features

    # Word lists
    p1s = {"i", "me", "my", "mine", "myself"}
    p1p = {"we", "us", "our", "ours", "ourselves"}
    p2 = {"you", "your", "yours", "yourself", "yourselves"}
    p3 = {
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "it",
        "its",
    }

    hedges = {
        "may",
        "might",
        "could",
        "would",
        "perhaps",
        "possibly",
        "probably",
        "likely",
        "somewhat",
        "suggest",
        "indicate",
        "seem",
        "appear",
        "approximately",
        "roughly",
    }
    boosters = {
        "clearly",
        "definitely",
        "obviously",
        "certainly",
        "always",
        "never",
        "proves",
        "demonstrates",
        "undoubtedly",
        "absolutely",
    }
    discourse = {
        "however",
        "therefore",
        "furthermore",
        "moreover",
        "nevertheless",
        "consequently",
        "although",
        "meanwhile",
        "thus",
        "hence",
        "indeed",
        "specifically",
    }

    words_lower = [w.lower() for w in doc.tokens]
    n_words = len(words_lower)
    per_1k = 1000.0 / n_words if n_words > 0 else 0.0

    p1s_count = sum(1 for w in words_lower if w in p1s)
    p1p_count = sum(1 for w in words_lower if w in p1p)
    p2_count = sum(1 for w in words_lower if w in p2)
    p3_count = sum(1 for w in words_lower if w in p3)

    hedge_count = sum(1 for w in words_lower if w in hedges)
    booster_count = sum(1 for w in words_lower if w in boosters)
    discourse_count = sum(1 for w in words_lower if w in discourse)

    features["pron_first_singular"] = p1s_count * per_1k
    features["pron_first_plural"] = p1p_count * per_1k
    features["pron_second"] = p2_count * per_1k
    features["pron_third"] = p3_count * per_1k

    features["self_immersion_index"] = p1s_count / (p1s_count + p1p_count + 1.0)
    total_pronouns = p1s_count + p1p_count + p2_count + p3_count
    features["audience_engagement_index"] = p2_count / (total_pronouns + 1.0)

    features["hedge_ratio"] = hedge_count * per_1k
    features["booster_ratio"] = booster_count * per_1k
    features["hedge_booster_ratio"] = hedge_count / (booster_count + 1.0)
    features["discourse_marker_ratio"] = discourse_count * per_1k

    # Contractions
    contraction_count = len(_CONTRACTION_PATTERN.findall(doc.cleaned_text))
    features["contraction_ratio"] = contraction_count * per_1k

    # Formality Score (Heylighen & Dewaele, 1999)
    spacy_doc = doc.spacy_doc
    total_pos = sum(1 for token in spacy_doc if not token.is_punct and not token.is_space)
    if total_pos > 0:
        pos_counts = {
            "NOUN": 0,
            "PROPN": 0,
            "ADJ": 0,
            "ADP": 0,
            "DET": 0,
            "PRON": 0,
            "VERB": 0,
            "ADV": 0,
            "INTJ": 0,
        }
        for token in spacy_doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1

        f1 = (
            (
                pos_counts["NOUN"]
                + pos_counts["PROPN"]
                + pos_counts["ADJ"]
                + pos_counts["ADP"]
                + pos_counts["DET"]
            )
            / total_pos
            * 100
        )
        f2 = (
            (pos_counts["PRON"] + pos_counts["VERB"] + pos_counts["ADV"] + pos_counts["INTJ"])
            / total_pos
            * 100
        )
        features["formality_score"] = (f1 - f2 + 100) / 2.0

    # Sentiment
    analyzer = _get_vader_analyzer()
    sentiments = []
    for sent in doc.sentences:
        score = analyzer.polarity_scores(sent)["compound"]
        sentiments.append(score)

    if sentiments:
        features["sentiment_mean"] = float(np.mean(sentiments))
        features["sentiment_std"] = float(np.std(sentiments))
        features["sentiment_positive_ratio"] = sum(1 for s in sentiments if s > 0.05) / len(
            sentiments
        )
        features["sentiment_negative_ratio"] = sum(1 for s in sentiments if s < -0.05) / len(
            sentiments
        )

    return features
