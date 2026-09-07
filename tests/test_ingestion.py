import pytest
from pathlib import Path
from idiolect.ingestion import clean_text, ingest
from idiolect.models import Document

def test_clean_text():
    dirty = "This  is\t\ta   test\n\n\n\nwith   spaces."
    cleaned = clean_text(dirty)
    assert "  " not in cleaned
    assert "\t" not in cleaned
    assert cleaned == "This is a test\n\nwith spaces."

def test_ingest():
    sample = "The quick brown fox jumps over the lazy dog. It was a very agile fox!"
    doc = ingest(sample)
    assert isinstance(doc, Document)
    assert doc.word_count > 0
    assert doc.sentence_count == 2
    assert len(doc.tokens) > 0
    assert doc.spacy_doc is not None
