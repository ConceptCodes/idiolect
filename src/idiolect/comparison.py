"""Compares two linguistic Fingerprints."""

from __future__ import annotations

import math
import numpy as np

try:
    from scipy.spatial.distance import cosine
except ImportError:
    # Fallback if scipy is not installed
    def cosine(u, v):
        uv = np.dot(u, v)
        uu = np.dot(u, u)
        vv = np.dot(v, v)
        if uu == 0 or vv == 0:
            return 1.0
        return 1.0 - uv / math.sqrt(uu * vv)

from .models import ComparisonResult, Fingerprint
# Import POPULATION_STATS to use for Burrows' Delta z-scoring
from .fingerprint import POPULATION_STATS


def compare(fp_a: Fingerprint, fp_b: Fingerprint) -> ComparisonResult:
    """Compare two fingerprints and generate a ComparisonResult.
    
    Calculates cosine similarity, Burrows' Delta, per-axis differences,
    and identifies the most similar and most divergent features.
    """
    # 1. Extract feature vectors
    features_a = fp_a.features
    features_b = fp_b.features
    
    # 2. Extract calibrated features in POPULATION_STATS for standardized comparison
    z_a_list = []
    z_b_list = []
    feature_diffs = []
    manhattan_sum = 0.0
    valid_features = 0

    for k, (mean, std) in POPULATION_STATS.items():
        if k in features_a and k in features_b and std > 0:
            za = (features_a[k] - mean) / std
            zb = (features_b[k] - mean) / std
            z_a_list.append(za)
            z_b_list.append(zb)
            diff = abs(za - zb)
            manhattan_sum += diff
            valid_features += 1
            feature_diffs.append((k, diff))

    # Also compare any remaining common features
    for k in features_a:
        if k in features_b and k not in POPULATION_STATS:
            diff = abs(features_a[k] - features_b[k])
            feature_diffs.append((k, diff))

    # 3. Standardized vectors
    if valid_features > 0:
        za_vec = np.array(z_a_list, dtype=float)
        zb_vec = np.array(z_b_list, dtype=float)
        
        norm_a = np.linalg.norm(za_vec)
        norm_b = np.linalg.norm(zb_vec)
        
        if norm_a > 0 and norm_b > 0:
            # Cosine similarity in z-score space
            z_cos_sim = float(np.dot(za_vec, zb_vec) / (norm_a * norm_b))
        else:
            z_cos_sim = 0.0
            
        manhattan_delta = manhattan_sum / valid_features
    else:
        z_cos_sim = 0.0
        manhattan_delta = 2.0

    # 4. Compute per-axis deltas
    axis_deltas = {}
    axes_a = fp_a.axes
    axes_b = fp_b.axes
    common_axes = set(axes_a.keys()).intersection(axes_b.keys())
    
    for ax in common_axes:
        axis_deltas[ax] = abs(axes_a[ax] - axes_b[ax])
        
    mean_axis_delta = float(np.mean(list(axis_deltas.values()))) if axis_deltas else 20.0

    # Overall similarity score (0.0 to 1.0)
    # Combines Manhattan Delta (Burrows) and Axis divergence
    # Manhattan delta of 0 -> 1.0; delta of 2.2+ -> 0.0
    delta_score = max(0.0, min(1.0, 1.0 - (manhattan_delta / 2.2)))
    axis_score = max(0.0, min(1.0, 1.0 - (mean_axis_delta / 35.0)))
    
    cosine_similarity = 0.60 * delta_score + 0.40 * axis_score
    cosine_delta = 1.0 - max(0.0, z_cos_sim)

    # 5. Identify 5 most similar and 5 most divergent features
    feature_diffs.sort(key=lambda x: x[1])
    most_similar_features = [k for k, _ in feature_diffs[:5]]
    most_divergent_features = [k for k, _ in feature_diffs[-5:]]
    most_divergent_features.reverse()  # Largest difference first

    # 6. Determine same_author_likelihood (stylometric decision boundary)
    if manhattan_delta <= 0.80 and mean_axis_delta <= 9.0:
        same_author = "very_likely"
    elif manhattan_delta <= 1.15 and mean_axis_delta <= 15.0:
        same_author = "likely"
    elif manhattan_delta <= 1.55 and mean_axis_delta <= 22.0:
        same_author = "uncertain"
    elif manhattan_delta <= 1.95:
        same_author = "unlikely"
    else:
        same_author = "very_unlikely"

    return ComparisonResult(
        fingerprint_a=fp_a.label,
        fingerprint_b=fp_b.label,
        cosine_similarity=cosine_similarity,
        cosine_delta=cosine_delta,
        manhattan_delta=manhattan_delta,
        axis_deltas=axis_deltas,
        same_author_likelihood=same_author,
        most_similar_features=most_similar_features,
        most_divergent_features=most_divergent_features
    )
