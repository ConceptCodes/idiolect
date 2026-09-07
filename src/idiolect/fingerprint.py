"""Orchestrates feature extraction and synthesizes the Fingerprint."""

from __future__ import annotations

from pathlib import Path

from .features.character import extract_character
from .features.lexical import extract_lexical
from .features.pragmatic import extract_pragmatic
from .features.readability import extract_readability
from .features.syntactic import extract_syntactic
from .ingestion import ingest, ingest_file
from .models import AuthorType, Document, FeatureSet, Fingerprint

# --- Population Stats Hardcoded ---
# Used for z-score normalization and scaling across linguistic strata.
POPULATION_STATS = {
    # Lexical
    "lexical.mattr": (0.75, 0.10),
    "lexical.mtld": (70.0, 25.0),
    "lexical.hdd": (0.80, 0.08),
    "lexical.yule_k": (120.0, 60.0),
    "lexical.hapax_legomena_ratio": (0.50, 0.15),
    "lexical.rare_word_ratio": (0.12, 0.06),
    "lexical.avg_word_length": (4.7, 0.6),
    # Syntactic
    "syntactic.sent_length_mean": (16.0, 6.0),
    "syntactic.sent_length_cv": (0.55, 0.20),
    "syntactic.parse_tree_depth_mean": (5.0, 1.5),
    "syntactic.subordinate_clause_ratio": (0.25, 0.12),
    "syntactic.dep_arc_length_mean": (2.5, 0.6),
    "syntactic.pos_NOUN": (0.20, 0.05),
    "syntactic.pos_VERB": (0.13, 0.04),
    "syntactic.pos_ADJ": (0.08, 0.03),
    "syntactic.pos_ADP": (0.12, 0.03),
    "syntactic.pos_DET": (0.10, 0.03),
    # Readability
    "readability.flesch_reading_ease": (60.0, 18.0),
    "readability.flesch_kincaid_grade": (9.5, 3.5),
    # Pragmatic
    "pragmatic.formality_score": (55.0, 12.0),
    "pragmatic.hedge_booster_ratio": (1.5, 1.0),
    "pragmatic.sentiment_std": (0.25, 0.12),
    "pragmatic.sentiment_positive_ratio": (0.30, 0.15),
    "pragmatic.sentiment_negative_ratio": (0.20, 0.12),
    "pragmatic.pron_first_singular": (8.0, 8.0),
    "pragmatic.pron_first_plural": (6.0, 6.0),
    "pragmatic.pron_second": (12.0, 12.0),  # per 1k words
    "pragmatic.pron_third": (20.0, 12.0),
    "pragmatic.discourse_marker_ratio": (10.0, 6.0),  # per 1k words
    "pragmatic.hedge_ratio": (15.0, 8.0),  # per 1k words
    "pragmatic.contraction_ratio": (16.0, 12.0),  # per 1k words
    # Character & Structure
    "character.punct_comma": (50.0, 20.0),
    "character.punct_semicolon": (3.0, 4.0),
    "character.punct_colon": (2.0, 2.5),
    "character.punct_to_word_ratio": (0.12, 0.04),
    "character.question_sentence_ratio": (0.06, 0.06),
    "character.char_diversity": (0.60, 0.15),
}


def _get_z(val: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (val - mean) / std


def _scale_0_100(val: float, mean: float, std: float) -> float:
    z = _get_z(val, mean, std)
    # Map z=-3 to 0, z=+3 to 100
    s = (z + 3) / 6 * 100
    return max(0.0, min(100.0, s))


def create_fingerprint(doc: Document, label: str) -> Fingerprint:
    """Synthesize a Fingerprint from a Document."""
    feature_set = FeatureSet(
        lexical=extract_lexical(doc),
        syntactic=extract_syntactic(doc),
        readability=extract_readability(doc),
        pragmatic=extract_pragmatic(doc),
        character=extract_character(doc),
    )

    flat = feature_set.all_features()

    # 1. Lexical Richness
    lex_mattr = _scale_0_100(flat.get("lexical.mattr", 0.75), *POPULATION_STATS["lexical.mattr"])
    lex_mtld = _scale_0_100(flat.get("lexical.mtld", 70.0), *POPULATION_STATS["lexical.mtld"])
    lex_hdd = _scale_0_100(flat.get("lexical.hdd", 0.80), *POPULATION_STATS["lexical.hdd"])
    lex_rare = _scale_0_100(
        flat.get("lexical.rare_word_ratio", 0.15), *POPULATION_STATS["lexical.rare_word_ratio"]
    )
    lexical_richness = (lex_mattr + lex_mtld + lex_hdd + lex_rare) / 4

    # 2. Syntactic Complexity
    syn_len = _scale_0_100(
        flat.get("syntactic.sent_length_mean", 15.0),
        *POPULATION_STATS["syntactic.sent_length_mean"],
    )
    syn_depth = _scale_0_100(
        flat.get("syntactic.parse_tree_depth_mean", 5.0),
        *POPULATION_STATS["syntactic.parse_tree_depth_mean"],
    )
    syn_sub = _scale_0_100(
        flat.get("syntactic.subordinate_clause_ratio", 0.3),
        *POPULATION_STATS["syntactic.subordinate_clause_ratio"],
    )
    syn_arc = _scale_0_100(
        flat.get("syntactic.dep_arc_length_mean", 2.5),
        *POPULATION_STATS["syntactic.dep_arc_length_mean"],
    )
    syntactic_complexity = (syn_len + syn_depth + syn_sub + syn_arc) / 4

    # 3. Formality
    formality = max(0.0, min(100.0, flat.get("pragmatic.formality_score", 50.0)))

    # 4. Epistemic Stance (higher ratio = more cautious)
    epistemic_stance = _scale_0_100(
        flat.get("pragmatic.hedge_booster_ratio", 1.0),
        *POPULATION_STATS["pragmatic.hedge_booster_ratio"],
    )

    # 5. Pacing Cadence
    pace_cv = _scale_0_100(
        flat.get("syntactic.sent_length_cv", 0.5), *POPULATION_STATS["syntactic.sent_length_cv"]
    )
    pace_punct = _scale_0_100(
        flat.get("character.char_diversity", 0.6), *POPULATION_STATS["character.char_diversity"]
    )
    pacing_cadence = (pace_cv + pace_punct) / 2

    # 6. Affective Intensity
    aff_std = _scale_0_100(
        flat.get("pragmatic.sentiment_std", 0.15), *POPULATION_STATS["pragmatic.sentiment_std"]
    )
    aff_pos = _scale_0_100(
        flat.get("pragmatic.sentiment_positive_ratio", 0.3),
        *POPULATION_STATS["pragmatic.sentiment_positive_ratio"],
    )
    aff_neg = _scale_0_100(
        flat.get("pragmatic.sentiment_negative_ratio", 0.2),
        *POPULATION_STATS["pragmatic.sentiment_negative_ratio"],
    )
    affective_intensity = (aff_std + aff_pos + aff_neg) / 3

    # 7. Interactive Engagement
    int_pron = _scale_0_100(
        flat.get("pragmatic.pron_second", 12.0), *POPULATION_STATS["pragmatic.pron_second"]
    )
    int_disc = _scale_0_100(
        flat.get("pragmatic.discourse_marker_ratio", 10.0),
        *POPULATION_STATS["pragmatic.discourse_marker_ratio"],
    )
    int_quest = _scale_0_100(
        flat.get("character.question_sentence_ratio", 0.06),
        *POPULATION_STATS["character.question_sentence_ratio"],
    )
    interactive_engagement = (int_pron + int_disc + int_quest) / 3

    axes = {
        "lexical_richness": lexical_richness,
        "syntactic_complexity": syntactic_complexity,
        "formality": formality,
        "epistemic_stance": epistemic_stance,
        "pacing_cadence": pacing_cadence,
        "affective_intensity": affective_intensity,
        "interactive_engagement": interactive_engagement,
    }

    # Standout traits
    standout_traits = []
    for k, v in flat.items():
        if k in POPULATION_STATS:
            mean, std = POPULATION_STATS[k]
            z = _get_z(v, mean, std)
            if abs(z) > 1.5:
                desc = f"Significantly {'higher' if z > 0 else 'lower'} than average."
                standout_traits.append(
                    {"feature": k, "value": v, "z_score": z, "interpretation": desc}
                )
    standout_traits.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    # AI Detection Indicators (0.0 = distinctly Human, 1.0 = distinctly AI)
    # 1. Sentence length uniformity (AI has very low CV / uniform length)
    sent_length_cv = flat.get("syntactic.sent_length_cv", 0.55)
    sent_length_uniformity = max(0.0, min(1.0, (0.60 - sent_length_cv) / 0.35))

    # 2. Emotional / Sentiment Flatness (AI text tends to be emotionally neutral)
    sentiment_std = flat.get("pragmatic.sentiment_std", 0.25)
    sentiment_flatness = max(0.0, min(1.0, (0.30 - sentiment_std) / 0.22))

    # 3. Contraction absence (AI standard outputs avoid contractions)
    contraction_ratio = flat.get("pragmatic.contraction_ratio", 16.0)
    contraction_absence = max(0.0, min(1.0, (20.0 - contraction_ratio) / 18.0))

    # 4. Impersonal voice (AI avoids first/second person personal pronouns)
    p1_singular = flat.get("pragmatic.pron_first_singular", 0.0)
    p2_second = flat.get("pragmatic.pron_second", 0.0)
    personal_pronouns = p1_singular + p2_second
    impersonal_voice = max(0.0, min(1.0, (22.0 - personal_pronouns) / 20.0))

    # 5. Expressive punctuation absence (Human writing uses dashes, ?, !)
    q_ratio = flat.get("character.question_sentence_ratio", 0.0)
    excl_ratio = flat.get("character.exclamation_sentence_ratio", 0.0)
    dash_rate = flat.get("character.punct_dash", 0.0)
    expressive_punct = (q_ratio + excl_ratio) * 100.0 + dash_rate
    expressive_punct_absence = max(0.0, min(1.0, (12.0 - expressive_punct) / 10.0))

    # 6. Formal transition marker density (AI frequently uses furthermore, moreover, consequently)
    disc_marker = flat.get("pragmatic.discourse_marker_ratio", 10.0)
    discourse_predictability = max(0.0, min(1.0, (disc_marker - 6.0) / 16.0))

    ai_indicators = {
        "sent_length_uniformity": sent_length_uniformity,
        "sentiment_flatness": sentiment_flatness,
        "contraction_absence": contraction_absence,
        "impersonal_voice": impersonal_voice,
        "expressive_punct_absence": expressive_punct_absence,
        "discourse_predictability": discourse_predictability,
    }

    # Weighted average: sentence uniformity, contractions, and personal voice are strongest signals
    weights = {
        "sent_length_uniformity": 0.25,
        "sentiment_flatness": 0.15,
        "contraction_absence": 0.25,
        "impersonal_voice": 0.20,
        "expressive_punct_absence": 0.10,
        "discourse_predictability": 0.05,
    }

    ai_score = sum(ai_indicators[k] * weights[k] for k in ai_indicators)
    if ai_score >= 0.60:
        author_type = AuthorType.AI
    elif ai_score <= 0.40:
        author_type = AuthorType.HUMAN
    else:
        author_type = AuthorType.UNCERTAIN

    return Fingerprint(
        label=label,
        word_count=doc.word_count,
        sentence_count=doc.sentence_count,
        source_path=doc.source_path,
        axes=axes,
        features=flat,
        standout_traits=standout_traits,
        ai_indicators=ai_indicators,
        author_type=author_type,
        ai_confidence=min(1.0, abs(ai_score - 0.5) * 2.2),
    )


def create_fingerprint_from_text(text: str, label: str | None = None) -> Fingerprint:
    """Ingest raw text and synthesize a Fingerprint."""
    doc = ingest(text)
    return create_fingerprint(doc, label=label or "text")


def create_fingerprint_from_file(path: Path, label: str | None = None) -> Fingerprint:
    """Ingest a text file and synthesize a Fingerprint."""
    doc = ingest_file(path)
    return create_fingerprint(doc, label=label or path.name)
