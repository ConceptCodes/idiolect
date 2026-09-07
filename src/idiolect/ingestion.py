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


def extract_text_from_docx(path: Path) -> str:
    """Extract text from a Microsoft Word (.docx) document.

    Extracts paragraphs and table rows, preserving structural boundaries.
    Falls back to direct XML parsing if python-docx parsing encounters issues.
    """
    try:
        import docx

        doc = docx.Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))
        if paragraphs:
            return "\n\n".join(paragraphs)
    except Exception:
        pass

    # Fallback to standard library zipfile + elementtree extraction
    import xml.etree.ElementTree as ET
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read("word/document.xml")
        root = ET.fromstring(xml_content)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paras = []
        for p in root.findall(".//w:p", ns):
            texts = [node.text for node in p.findall(".//w:t", ns) if node.text]
            if texts:
                paras.append("".join(texts))
        return "\n\n".join(paras)
    except Exception as e:
        raise ValueError(f"Failed to extract text from Word document {path}: {e}")


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF document using pypdf.

    Preserves page breaks as paragraph dividers.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                pages_text.append(t.strip())
        if not pages_text:
            raise ValueError(f"No extractable text found in PDF: {path}")
        return "\n\n".join(pages_text)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF document {path}: {e}")


def load_text(path: Path) -> str:
    """Load text from a file path.

    Supports plain text (.txt, .md, .text, .markdown, .rst),
    Microsoft Word (.docx), and PDF (.pdf) documents.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".txt", ".md", ".text", ".markdown", ".rst"):
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".docx":
        return extract_text_from_docx(path)
    elif suffix == ".pdf":
        return extract_text_from_pdf(path)
    else:
        # Fallback to plain text read with error replacement
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="replace")


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
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]

    # All tokens including punctuation (but not whitespace)
    all_tokens = [token.text for token in doc if not token.is_space]

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


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".rst",
    ".docx",
    ".pdf",
}


def find_text_files(path: Path) -> list[Path]:
    """Find all supported documents from a single file or directory.

    Supported extensions: .txt, .md, .text, .markdown, .rst, .docx, .pdf.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = [
            p
            for p in path.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and (p.suffix.lower() in SUPPORTED_EXTENSIONS or not p.suffix)
        ]
        return sorted(files, key=lambda p: p.name.lower())
    return []


def ingest_path(path: Path) -> Document:
    """Load and ingest a file or a directory of files into a Document.

    If a directory is provided, text from all contained text files
    is combined into a single unified Document.

    Args:
        path: Path to a file or directory.

    Returns:
        A fully populated Document ready for feature extraction.
    """
    path = Path(path)
    if path.is_file():
        return ingest_file(path)
    elif path.is_dir():
        files = find_text_files(path)
        if not files:
            raise ValueError(f"No readable text files found in directory: {path}")
        combined_texts = [load_text(f) for f in files]
        combined_raw = "\n\n".join(combined_texts)
        return ingest(combined_raw, source_path=str(path))
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")
