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


def test_find_text_files_and_ingest_path(tmp_path):
    from idiolect.ingestion import find_text_files, ingest_path

    # Directory with multiple supported text files and ignored files
    dir_path = tmp_path / "corpus"
    dir_path.mkdir()

    f1 = dir_path / "chapter1.txt"
    f1.write_text("The journey began early in the crisp morning air.")

    f2 = dir_path / "chapter2.md"
    f2.write_text("Across the wide valley, shadows grew as night approached.")

    hidden = dir_path / ".hidden.txt"
    hidden.write_text("Should be ignored.")

    ignored = dir_path / "data.csv"
    ignored.write_text("col1,col2\n1,2")

    found = find_text_files(dir_path)
    assert len(found) == 2
    assert found[0].name == "chapter1.txt"
    assert found[1].name == "chapter2.md"

    # Single file
    assert find_text_files(f1) == [f1]

    # Non-existent
    assert find_text_files(tmp_path / "does_not_exist") == []

    # Ingest path on directory
    combined_doc = ingest_path(dir_path)
    assert "The journey began early" in combined_doc.cleaned_text
    assert "Across the wide valley" in combined_doc.cleaned_text
    assert combined_doc.sentence_count >= 2

    # Ingest path on single file
    single_doc = ingest_path(f1)
    assert "The journey began early" in single_doc.cleaned_text
