"""Stylometric attribution explainability and document length confidence damping.

Provides:
- Proportional confidence damping and length warning utilities for short texts (<250 words).
- Aligning trait discovery to explain which specific linguistic markers drove an attribution.
"""

from __future__ import annotations

from typing import Any

from .fingerprint import POPULATION_STATS
from .models import Fingerprint

MIN_RELIABLE_WORDS = 250
SHORT_DOC_WARNING = (
    "Short document (<250 words): stylometric variance may reduce attribution confidence."
)


def compute_length_damping(word_count: int, min_words: int = MIN_RELIABLE_WORDS) -> float:
    """Compute proportional confidence damping factor for short documents.

    Stylometric statistics (such as MATTR, MTLD, and syntactic parse depth variance)
    suffer from high sampling noise when texts have fewer than 250 words.
    Returns 1.0 for texts >= 250 words, scaling down proportionally via
    sqrt(words / min_words) with a floor of 0.40.
    """
    if word_count >= min_words:
        return 1.0
    if word_count <= 0:
        return 0.40
    factor = (word_count / min_words) ** 0.5
    return max(0.40, min(1.0, factor))


# Human-readable linguistic trait interpretations based on z-score direction
TRAIT_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    # (feature_key): (Human Name, high_description, low_description)
    "character.punct_semicolon": (
        "Semicolon Usage",
        "high semicolon frequency",
        "infrequent semicolon usage",
    ),
    "pragmatic.contraction_ratio": (
        "Contraction Rate",
        "frequent contraction usage (informal tone)",
        "low contraction rate (formal register)",
    ),
    "syntactic.subordinate_clause_ratio": (
        "Subordinate Clauses",
        "complex subordinate clause structure",
        "direct coordinate clause structure",
    ),
    "syntactic.sent_length_mean": (
        "Sentence Length",
        "lengthy, expansive sentence constructions",
        "concise sentence length rhythm",
    ),
    "syntactic.sent_length_cv": (
        "Sentence Length Variance",
        "high sentence length variation (rhythmic pacing)",
        "uniform, measured sentence lengths",
    ),
    "syntactic.parse_tree_depth_mean": (
        "Syntactic Tree Depth",
        "deep syntactic clause nesting",
        "shallow, direct syntactic phrasing",
    ),
    "syntactic.dep_arc_length_mean": (
        "Dependency Arc Span",
        "long dependency spans (complex phrasal binding)",
        "compact dependency spans (tight phrasal binding)",
    ),
    "lexical.mattr": (
        "Vocabulary Richness (MATTR)",
        "high moving-average lexical diversity",
        "focused, repetitive core vocabulary",
    ),
    "lexical.mtld": (
        "Lexical Diversity (MTLD)",
        "sustained vocabulary diversity across paragraphs",
        "rapid vocabulary repetition",
    ),
    "lexical.rare_word_ratio": (
        "Sophisticated Lexicon",
        "elevated sophisticated/rare word density",
        "plain, accessible vocabulary",
    ),
    "lexical.hapax_legomena_ratio": (
        "Unique Word Ratio",
        "high proportion of single-occurrence words",
        "predominantly shared recurrent vocabulary",
    ),
    "lexical.avg_word_length": (
        "Word Length",
        "polysyllabic, sophisticated word length",
        "short, compact word forms",
    ),
    "pragmatic.hedge_booster_ratio": (
        "Epistemic Hedging",
        "cautious epistemic hedging",
        "direct, assertive epistemic stance",
    ),
    "pragmatic.formality_score": (
        "Register Formality",
        "elevated academic/formal register",
        "conversational, informal register",
    ),
    "pragmatic.pron_first_singular": (
        "First-Person Pronouns",
        "frequent first-person singular perspective ('I')",
        "impersonal perspective (minimal 'I')",
    ),
    "pragmatic.pron_first_plural": (
        "First-Person Plural",
        "inclusive first-person plural framing ('we')",
        "absence of first-person plural framing",
    ),
    "pragmatic.pron_second": (
        "Second-Person Direct Address",
        "frequent direct audience address ('you')",
        "neutral, third-party framing (no 'you')",
    ),
    "pragmatic.sentiment_std": (
        "Emotional Volatility",
        "dynamic emotional and affective variance",
        "steady, emotionally neutral tone",
    ),
    "pragmatic.discourse_marker_ratio": (
        "Discourse Transitions",
        "frequent transitional discourse markers",
        "compact transitions without discourse markers",
    ),
    "character.punct_comma": (
        "Comma Density",
        "heavy comma punctuation density",
        "light, sparse comma usage",
    ),
    "character.punct_colon": (
        "Colon Punctuation",
        "frequent colon elaborations",
        "infrequent colon usage",
    ),
    "character.question_sentence_ratio": (
        "Rhetorical Questions",
        "frequent interrogative/rhetorical questions",
        "declarative phrasing without rhetorical questions",
    ),
    "readability.flesch_reading_ease": (
        "Reading Ease",
        "accessible reading ease",
        "dense, scholarly reading level",
    ),
    "readability.flesch_kincaid_grade": (
        "Grade Level",
        "advanced collegiate reading grade level",
        "accessible introductory grade level",
    ),
    "syntactic.pos_NOUN": (
        "Noun Density",
        "dense nominal phrasing",
        "sparse nominal phrasing",
    ),
    "syntactic.pos_VERB": (
        "Verb Density",
        "action-oriented verbal density",
        "stative phrasing with fewer verbs",
    ),
    "syntactic.pos_ADJ": (
        "Adjective Density",
        "descriptive modifier richness",
        "unadorned phrasing with few modifiers",
    ),
}


def explain_aligning_traits(
    cand_fp: Fingerprint,
    essay_fp: Fingerprint,
    top_n: int = 4,
) -> list[dict[str, Any]]:
    """Identify the top linguistic traits that drove the attribution match.

    Compares z-scores in standardized population space, scoring candidates by:
    - Closeness of the two fingerprints on the trait (|z_a - z_b|)
    - Distinctiveness of the shared trait (|z_mean|)

    Returns a list of dictionaries with feature name, description, and delta.
    """
    cand_feats = cand_fp.features
    essay_feats = essay_fp.features

    candidate_alignments = []

    for k, (mean, std) in POPULATION_STATS.items():
        if k in cand_feats and k in essay_feats and std > 0:
            za = (cand_feats[k] - mean) / std
            zb = (essay_feats[k] - mean) / std
            diff = abs(za - zb)
            z_mean = (za + zb) / 2.0

            salience_score = diff - 0.25 * min(2.0, abs(z_mean))

            if k in TRAIT_DESCRIPTIONS:
                human_name, high_desc, low_desc = TRAIT_DESCRIPTIONS[k]
                description = high_desc if z_mean >= 0 else low_desc
            else:
                clean_k = k.split(".", 1)[-1].replace("_", " ").title()
                human_name = clean_k
                description = (
                    f"elevated {clean_k.lower()}" if z_mean >= 0 else f"subdued {clean_k.lower()}"
                )

            candidate_alignments.append(
                {
                    "feature": k,
                    "name": human_name,
                    "description": description,
                    "z_delta": round(diff, 3),
                    "candidate_val": round(cand_feats[k], 3),
                    "essay_val": round(essay_feats[k], 3),
                    "salience_score": salience_score,
                }
            )

    candidate_alignments.sort(key=lambda x: x["salience_score"])

    results = []
    for item in candidate_alignments[:top_n]:
        results.append(
            {
                "feature": item["feature"],
                "name": item["name"],
                "description": item["description"],
                "z_delta": item["z_delta"],
                "candidate_val": item["candidate_val"],
                "essay_val": item["essay_val"],
            }
        )

    return results


def format_aligning_traits_summary(traits: list[dict[str, Any]]) -> str:
    """Format aligning traits as a semicolon-separated summary string."""
    return "; ".join(t["description"] for t in traits)
