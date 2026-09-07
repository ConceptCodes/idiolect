"""Core data models for the idiolect fingerprinting system.

Defines the contracts between ingestion, feature extraction,
fingerprint synthesis, comparison, and reporting modules.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class AuthorType(str, Enum):
    """Classification of the likely author type."""

    HUMAN = "human"
    AI = "ai"
    UNCERTAIN = "uncertain"


@dataclass
class Document:
    """A preprocessed text document ready for feature extraction.

    This is the input contract for all feature extractors.
    The spaCy Doc object is attached after ingestion.
    """

    raw_text: str
    """Original text exactly as provided."""

    cleaned_text: str
    """Normalized text (Unicode cleaned, whitespace normalized)."""

    sentences: list[str]
    """List of sentence strings extracted from the text."""

    tokens: list[str]
    """List of token strings (words only, no punctuation)."""

    all_tokens: list[str]
    """List of all token strings including punctuation."""

    word_count: int
    """Total number of word tokens."""

    sentence_count: int
    """Total number of sentences."""

    source_path: str | None = None
    """Path to source file, if loaded from disk."""

    # The spaCy Doc is not serializable; attached at runtime
    _spacy_doc: Any = field(default=None, repr=False)

    @property
    def spacy_doc(self) -> Any:
        """Access the spaCy Doc object."""
        if self._spacy_doc is None:
            raise RuntimeError("spaCy Doc not attached. Run ingestion first.")
        return self._spacy_doc


@dataclass
class FeatureSet:
    """Raw extracted features organized by linguistic stratum.

    Each stratum maps feature names to float values.
    This is the output contract for feature extractors
    and input for fingerprint synthesis.
    """

    lexical: dict[str, float] = field(default_factory=dict)
    """Vocabulary richness, word frequency, hapax ratios."""

    syntactic: dict[str, float] = field(default_factory=dict)
    """Sentence structure, POS distributions, dependency patterns."""

    readability: dict[str, float] = field(default_factory=dict)
    """Reading ease, grade levels, complexity scores."""

    pragmatic: dict[str, float] = field(default_factory=dict)
    """Pronouns, hedging, formality, discourse markers."""

    character: dict[str, float] = field(default_factory=dict)
    """Punctuation patterns, casing, contractions."""

    def all_features(self) -> dict[str, float]:
        """Flatten all feature strata into a single dict."""
        combined: dict[str, float] = {}
        for stratum_name in ("lexical", "syntactic", "readability", "pragmatic", "character"):
            stratum = getattr(self, stratum_name)
            for key, value in stratum.items():
                combined[f"{stratum_name}.{key}"] = value
        return combined

    @property
    def total_feature_count(self) -> int:
        """Total number of features across all strata."""
        return sum(
            len(getattr(self, s))
            for s in ("lexical", "syntactic", "readability", "pragmatic", "character")
        )


# --- Fingerprint Axis Definitions ---

FINGERPRINT_AXES = [
    "lexical_richness",
    "syntactic_complexity",
    "formality",
    "epistemic_stance",
    "pacing_cadence",
    "affective_intensity",
    "interactive_engagement",
]

AXIS_DESCRIPTIONS = {
    "lexical_richness": "Vocabulary diversity, rare word usage, and word sophistication",
    "syntactic_complexity": (
        "Sentence structure variation, clause depth, and grammatical complexity"
    ),
    "formality": "Register formality, nominalization, and context-independence",
    "epistemic_stance": "Certainty vs. hedging, modal verb patterns, and evidentiality",
    "pacing_cadence": "Sentence length rhythm, punctuation density, and structural variety",
    "affective_intensity": "Emotional volatility, sentiment strength, and affective range",
    "interactive_engagement": "Audience address, discourse markers, and conversational style",
}


@dataclass
class Fingerprint:
    """A complete linguistic fingerprint for a text or author.

    Contains both the interpretable 7-axis radar profile
    and the full raw feature vector for comparison.
    """

    # Identity
    label: str
    """Human-readable label (author name or file name)."""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO 8601 timestamp of creation."""

    # Document metadata
    word_count: int = 0
    sentence_count: int = 0
    source_path: str | None = None

    # 7-Axis Radar Profile (0-100 percentile scores)
    axes: dict[str, float] = field(default_factory=dict)
    """Seven interpretable axes, each scored 0-100."""

    # Raw feature vector
    features: dict[str, float] = field(default_factory=dict)
    """Complete flattened feature vector for comparison."""

    # Standout traits (z-score deviations)
    standout_traits: list[dict[str, Any]] = field(default_factory=list)
    """Features with |z-score| > 1.5, sorted by magnitude."""

    # AI detection signals
    ai_indicators: dict[str, float] = field(default_factory=dict)
    """Signals that correlate with AI-generated content."""

    author_type: AuthorType = AuthorType.UNCERTAIN
    """Predicted author type classification."""

    ai_confidence: float = 0.0
    """Confidence in the author_type prediction (0-1)."""

    # Multi-sample profiling metadata
    sample_count: int = 1
    """Number of individual writing samples aggregated into this fingerprint."""

    axis_stability: dict[str, float] = field(default_factory=dict)
    """Standard deviation of 7-axis radar scores across samples (lower = more consistent)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        d["author_type"] = self.author_type.value
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fingerprint:
        """Deserialize from a dictionary."""
        data = data.copy()
        data["author_type"] = AuthorType(data.get("author_type", "uncertain"))
        data.setdefault("sample_count", 1)
        data.setdefault("axis_stability", {})
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> Fingerprint:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Path) -> None:
        """Save fingerprint to a JSON file."""
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> Fingerprint:
        """Load fingerprint from a JSON file."""
        return cls.from_json(path.read_text())


@dataclass
class AuthorSample:
    """A single writing sample enrolled for an author profile."""

    id: int | None
    author_label: str
    sample_label: str
    word_count: int
    enrolled_at: str
    fingerprint: Fingerprint


@dataclass
class AuthorProfile:
    """Aggregated stylometric profile for an author composed of multiple samples."""

    label: str
    composite_fingerprint: Fingerprint
    samples: list[AuthorSample] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples) if self.samples else self.composite_fingerprint.sample_count

    @property
    def total_word_count(self) -> int:
        if self.samples:
            return sum(s.word_count for s in self.samples)
        return self.composite_fingerprint.word_count


@dataclass
class ComparisonResult:
    """Result of comparing two fingerprints."""

    fingerprint_a: str
    """Label of the first fingerprint."""

    fingerprint_b: str
    """Label of the second fingerprint."""

    # Overall similarity
    cosine_similarity: float = 0.0
    """Cosine similarity between feature vectors (0-1)."""

    cosine_delta: float = 0.0
    """Cosine Delta distance (lower = more similar)."""

    manhattan_delta: float = 0.0
    """Burrows' Delta (Manhattan distance over z-scores)."""

    # Axis-level comparison
    axis_deltas: dict[str, float] = field(default_factory=dict)
    """Per-axis absolute difference."""

    # Interpretation
    same_author_likelihood: str = "uncertain"
    """Qualitative assessment: 'very_likely', 'likely', 'uncertain', 'unlikely', 'very_unlikely'."""

    most_similar_features: list[str] = field(default_factory=list)
    """Top features where the two fingerprints agree."""

    most_divergent_features: list[str] = field(default_factory=list)
    """Top features where the two fingerprints diverge."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)
