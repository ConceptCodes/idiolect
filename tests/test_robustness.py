from pathlib import Path

from typer.testing import CliRunner

import idiolect.store
from idiolect.cli import app
from idiolect.comparison import compare
from idiolect.fingerprint import create_fingerprint
from idiolect.ingestion import find_text_files, ingest
from idiolect.models import AuthorType
from idiolect.profiling import aggregate_fingerprints

runner = CliRunner()


def test_short_document_ai_detection_guard():
    # Empty string
    doc_empty = ingest("")
    fp_empty = create_fingerprint(doc_empty, "empty")
    assert fp_empty.author_type == AuthorType.UNCERTAIN
    assert fp_empty.ai_confidence == 0.0

    # Two words
    doc_short = ingest("Hello world.")
    fp_short = create_fingerprint(doc_short, "short")
    assert fp_short.author_type == AuthorType.UNCERTAIN
    assert fp_short.ai_confidence == 0.0

    # Composite profile with small word count
    composite = aggregate_fingerprints([fp_short], label="Alice")
    assert composite.author_type == AuthorType.UNCERTAIN
    assert composite.ai_confidence == 0.0


def test_phantom_divergent_features_identical_docs():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Several sentences are provided to ensure a standard text sample with varied vocabulary."
    )
    doc = ingest(text)
    fp1 = create_fingerprint(doc, "fp1")
    fp2 = create_fingerprint(doc, "fp2")

    comp = compare(fp1, fp2)
    assert comp.cosine_similarity == 1.0
    assert comp.manhattan_delta == 0.0
    assert comp.same_author_likelihood == "very_likely"
    # Identical documents should have NO divergent features
    assert comp.most_divergent_features == []


def test_recursive_find_text_files(tmp_path: Path):
    root = tmp_path / "submissions"
    sub1 = root / "student_01"
    sub2 = root / "student_02" / "drafts"
    hidden = root / ".git"

    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    hidden.mkdir(parents=True)

    (sub1 / "essay.txt").write_text("Essay 1 content")
    (sub2 / "essay.md").write_text("Essay 2 content")
    (root / "rubric.txt").write_text("Rubric content")
    (hidden / "config.txt").write_text("Hidden config")
    (sub1 / ".hidden_draft.txt").write_text("Hidden draft")

    files = find_text_files(root, recursive=True)
    filenames = [f.name for f in files]
    assert "essay.txt" in filenames
    assert "essay.md" in filenames
    assert "rubric.txt" in filenames
    assert "config.txt" not in filenames
    assert ".hidden_draft.txt" not in filenames

    # Non-recursive should only find top-level
    shallow_files = find_text_files(root, recursive=False)
    assert len(shallow_files) == 1
    assert shallow_files[0].name == "rubric.txt"


def test_batch_fault_tolerance(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "fault_tolerance.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    batch_dir = tmp_path / "batch_docs"
    batch_dir.mkdir()

    # Good document
    good_doc = batch_dir / "good.txt"
    good_doc.write_text(
        "Modern computational stylometry applies quantitative linguistic analysis to text corpora. "
        "Through lexical and syntactic feature extraction, individual idiolect markers emerge."
    )

    # Bad document: simulated corrupted PDF
    bad_pdf = batch_dir / "corrupted.pdf"
    bad_pdf.write_bytes(b"%PDF-corrupted-bytes-not-a-valid-pdf-structure")

    # 1. Batch Analyze should succeed on good.txt and skip bad_pdf
    res_a = runner.invoke(app, ["analyze", str(batch_dir), "--no-report"])
    assert res_a.exit_code == 0
    assert "good.txt" in res_a.output
    assert "Skipped 1 unreadable/corrupted file" in res_a.output

    # 2. Batch Enroll should succeed on good.txt and skip bad_pdf
    res_e = runner.invoke(app, ["enroll", "AuthorA", str(batch_dir)])
    assert res_e.exit_code == 0
    assert "Successfully enrolled author profile" in res_e.output
    assert "Skipped 1 unreadable file" in res_e.output

    # 3. Batch Verify should succeed on good.txt and skip bad_pdf
    res_v = runner.invoke(app, ["verify", "AuthorA", str(batch_dir)])
    assert res_v.exit_code == 0
    assert "good.txt" in res_v.output
    assert "Skipped 1 unreadable/corrupted file" in res_v.output

    # 4. Batch Identify should succeed on good.txt and skip bad_pdf
    res_i = runner.invoke(app, ["identify", str(batch_dir), "--no-report"])
    assert res_i.exit_code == 0
    assert "good.txt" in res_i.output
    assert "Skipped 1 unreadable file" in res_i.output


def test_store_batch_list_profiles_and_wal(tmp_path: Path):
    db_file = tmp_path / "test_wal.db"
    store = idiolect.store.FingerprintStore(db_path=db_file)

    # Enroll 2 authors with multiple samples
    text1 = "First sample text exploring narrative complexity and character voice."
    text2 = "Second sample text investigating philosophical inquiries and epistemology."
    doc1 = ingest(text1)
    doc2 = ingest(text2)
    fp1 = create_fingerprint(doc1, "s1")
    fp2 = create_fingerprint(doc2, "s2")

    store.enroll_sample("Alice", fp1, sample_label="essay1.txt")
    store.enroll_sample("Alice", fp2, sample_label="essay2.txt")

    store.enroll_sample("Bob", fp1, sample_label="bob_essay.txt")

    profiles = store.list_profiles()
    assert len(profiles) == 2
    prof_map = {p.label: p for p in profiles}

    assert "Alice" in prof_map
    assert prof_map["Alice"].sample_count == 2
    assert len(prof_map["Alice"].samples) == 2

    assert "Bob" in prof_map
    assert prof_map["Bob"].sample_count == 1
    assert len(prof_map["Bob"].samples) == 1
