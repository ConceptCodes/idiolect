"""Multi-sample profiling and weighted rolling average aggregation.

Synthesizes a representative, evolution-aware idiolect baseline across multiple
writing samples for an author. Weights individual samples by both volume (statistical power)
and recency (exponential rolling decay), while measuring intra-author stylistic stability.
"""

from __future__ import annotations

import numpy as np

from .fingerprint import POPULATION_STATS, _get_z
from .models import FINGERPRINT_AXES, AuthorType, Fingerprint


def calculate_sample_weights(
    sample_word_counts: list[int],
    recency_decay: float = 0.90,
    max_samples: int = 20,
) -> list[float]:
    """Calculate normalized weights for a sequence of samples.

    Samples are expected in chronological order (index 0 is oldest,
    index -1 is newest).

    The weight for each sample combines:
    1. Statistical volume: min(word_count, 5000), clamped at a minimum of 100 words.
    2. Recency decay: recency_decay^(age), where age=0 for the most recent sample.

    Args:
        sample_word_counts: List of word counts in chronological order.
        recency_decay: Exponential decay factor per step (default 0.90).
        max_samples: Maximum number of recent samples in the rolling window.

    Returns:
        List of normalized weights that sum to 1.0.
    """
    if not sample_word_counts:
        return []

    # Rolling window truncation
    counts = sample_word_counts[-max_samples:]
    n = len(counts)
    if n == 1:
        return [1.0]

    raw_weights = []
    for i, count in enumerate(counts):
        age = n - 1 - i
        recency = recency_decay**age
        vol = max(100, min(count, 5000))
        raw_weights.append(vol * recency)

    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / n] * n
    return [w / total for w in raw_weights]


def aggregate_fingerprints(
    fingerprints: list[Fingerprint],
    label: str,
    recency_decay: float = 0.90,
    max_samples: int = 20,
) -> Fingerprint:
    """Synthesize a composite Fingerprint from multiple sample Fingerprints.

    Uses a weighted rolling average based on sample volume and recency decay.

    Args:
        fingerprints: Sequence of Fingerprint objects in chronological order.
        label: The author label for the composite profile.
        recency_decay: Exponential decay factor for rolling average (0.1 to 1.0).
        max_samples: Rolling window size limit.

    Returns:
        Aggregated composite Fingerprint with axis stability metrics and sample count.
    """
    if not fingerprints:
        raise ValueError("Cannot aggregate an empty list of fingerprints.")

    if len(fingerprints) == 1:
        fp = fingerprints[0]
        return Fingerprint(
            label=label,
            created_at=fp.created_at,
            word_count=fp.word_count,
            sentence_count=fp.sentence_count,
            source_path=fp.source_path,
            axes=dict(fp.axes),
            features=dict(fp.features),
            standout_traits=list(fp.standout_traits),
            ai_indicators=dict(fp.ai_indicators),
            author_type=fp.author_type,
            ai_confidence=fp.ai_confidence,
            sample_count=1,
            axis_stability={ax: 0.0 for ax in FINGERPRINT_AXES},
        )

    # Rolling window of the most recent samples
    fps = fingerprints[-max_samples:]
    weights = calculate_sample_weights(
        [fp.word_count for fp in fps],
        recency_decay=recency_decay,
        max_samples=max_samples,
    )

    # 1. Aggregate features
    all_feature_keys: set[str] = set()
    for fp in fps:
        all_feature_keys.update(fp.features.keys())

    agg_features: dict[str, float] = {}
    for k in sorted(all_feature_keys):
        agg_features[k] = float(sum(w * fp.features.get(k, 0.0) for w, fp in zip(weights, fps)))

    # 2. Aggregate axes and compute stability (weighted standard deviation)
    agg_axes: dict[str, float] = {}
    axis_stability: dict[str, float] = {}
    for ax in FINGERPRINT_AXES:
        mean_ax = sum(w * fp.axes.get(ax, 50.0) for w, fp in zip(weights, fps))
        variance_ax = sum(
            w * ((fp.axes.get(ax, 50.0) - mean_ax) ** 2) for w, fp in zip(weights, fps)
        )
        agg_axes[ax] = float(max(0.0, min(100.0, mean_ax)))
        axis_stability[ax] = float(np.sqrt(max(0.0, variance_ax)))

    # 3. Aggregate AI indicators
    all_ind_keys: set[str] = set()
    for fp in fps:
        all_ind_keys.update(fp.ai_indicators.keys())

    agg_ai_indicators: dict[str, float] = {}
    for k in sorted(all_ind_keys):
        agg_ai_indicators[k] = float(
            sum(w * fp.ai_indicators.get(k, 0.0) for w, fp in zip(weights, fps))
        )

    # 4. Composite AI score & classification
    calibration_weights = {
        "sent_length_uniformity": 0.25,
        "sentiment_flatness": 0.15,
        "contraction_absence": 0.25,
        "impersonal_voice": 0.20,
        "expressive_punct_absence": 0.10,
        "discourse_predictability": 0.05,
    }
    ai_score = sum(
        agg_ai_indicators.get(k, 0.0) * calibration_weights.get(k, 0.0) for k in calibration_weights
    )
    if ai_score >= 0.60:
        author_type = AuthorType.AI
    elif ai_score <= 0.40:
        author_type = AuthorType.HUMAN
    else:
        author_type = AuthorType.UNCERTAIN

    ai_confidence = min(1.0, abs(ai_score - 0.5) * 2.2)

    # 5. Standout traits evaluated from aggregated features against POPULATION_STATS
    standout_traits = []
    for k, v in agg_features.items():
        if k in POPULATION_STATS:
            mean, std = POPULATION_STATS[k]
            z = _get_z(v, mean, std)
            if abs(z) > 1.5:
                desc = f"Significantly {'higher' if z > 0 else 'lower'} than average."
                standout_traits.append(
                    {"feature": k, "value": v, "z_score": z, "interpretation": desc}
                )
    standout_traits.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    total_words = sum(fp.word_count for fp in fps)
    total_sentences = sum(fp.sentence_count for fp in fps)

    return Fingerprint(
        label=label,
        word_count=total_words,
        sentence_count=total_sentences,
        source_path=None,
        axes=agg_axes,
        features=agg_features,
        standout_traits=standout_traits,
        ai_indicators=agg_ai_indicators,
        author_type=author_type,
        ai_confidence=ai_confidence,
        sample_count=len(fingerprints),
        axis_stability=axis_stability,
    )


def compute_profile_consistency(axis_stability: dict[str, float]) -> float:
    """Compute overall stylistic consistency percentage from axis standard deviations.

    Returns:
        Score between 0.0% (highly volatile) and 100.0% (exceptionally consistent).
    """
    if not axis_stability:
        return 100.0

    mean_sigma = float(np.mean(list(axis_stability.values())))
    # Mean sigma of 0 -> 100%; mean sigma of 25+ -> 12.5% or lower
    score = 100.0 - (mean_sigma * 3.5)
    return max(0.0, min(100.0, score))


def classify_stability(std_dev: float) -> str:
    """Qualitative classification of stylometric variance for an axis."""
    if std_dev <= 3.0:
        return "Highly Stable"
    if std_dev <= 6.0:
        return "Stable"
    if std_dev <= 10.0:
        return "Moderate Var."
    return "High Var."
