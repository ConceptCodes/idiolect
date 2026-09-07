"""Text ingestion and preprocessing.

Handles loading text from files (plain text, PDF future),
Unicode normalization, and spaCy processing to produce
a Document ready for feature extraction.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .models import Document


# Lazy-loaded spaCy model
_nlp = None


def _get_nlp():
    """Lazy-load the spaCy model to avoid startup cost when not needed."""
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_md")
        except OSError:
            # Fall back to small model
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy English model found. Install one with:\n"
                    "  python -m spacy download en_core_web_sm\n"
                    "  python -m spacy download en_core_web_md  (recommended)"
                )
        # Increase max length for longer documents
        _nlp.max_length = 2_000_000
    return _nlp


def clean_text(raw: str) -> str:
    """Normalize Unicode, fix encoding issues, and clean whitespace.

    Preserves punctuation patterns and casing (important for fingerprinting).
    """
    # Normalize Unicode (NFC form)
    text = unicodedata.normalize("NFC", raw)

    # Replace common Unicode variants with ASCII equivalents
    replacements = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "--",  # Em dash
        "\u2026": "...",  # Ellipsis
        "\u00a0": " ",  # Non-breaking space
        "\t": " ",  # Tab
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize whitespace runs (but preserve newlines for paragraph detection)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse multiple newlines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_text(path: Path) -> str:
    """Load text from a file path.

    Currently supports .txt files. PDF support will be added later.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".txt", ".md", ".text"):
        return path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        raise NotImplementedError(
            "PDF input is not yet supported. Convert to plain text first."
        )
    else:
        # Try reading as plain text
        return path.read_text(encoding="utf-8")


def ingest(text: str, source_path: str | None = None) -> Document:
    """Process raw text into a Document with spaCy annotations.

    Args:
        text: Raw input text.
        source_path: Optional path to the source file.

    Returns:
        A fully populated Document ready for feature extraction.
    """
    cleaned = clean_text(text)
    nlp = _get_nlp()
    doc = nlp(cleaned)

    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Extract word tokens (no punctuation, no whitespace)
    tokens = [
        token.text
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    # All tokens including punctuation (but not whitespace)
    all_tokens = [
        token.text
        for token in doc
        if not token.is_space
    ]

    document = Document(
        raw_text=text,
        cleaned_text=cleaned,
        sentences=sentences,
        tokens=tokens,
        all_tokens=all_tokens,
        word_count=len(tokens),
        sentence_count=len(sentences),
        source_path=source_path,
    )
    document._spacy_doc = doc

    return document


def ingest_file(path: Path) -> Document:
    """Load and ingest a text file into a Document.

    Args:
        path: Path to the text file.

    Returns:
        A fully populated Document ready for feature extraction.
    """
    raw = load_text(path)
    return ingest(raw, source_path=str(path))
