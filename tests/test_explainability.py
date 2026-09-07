import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import idiolect.store
from idiolect.cli import app
from idiolect.explainability import (
    SHORT_DOC_WARNING,
    compute_length_damping,
    explain_aligning_traits,
    format_aligning_traits_summary,
)
from idiolect.fingerprint import create_fingerprint
from idiolect.ingestion import ingest

runner = CliRunner()


def test_compute_length_damping():
    # Above threshold
    assert compute_length_damping(250) == 1.0
    assert compute_length_damping(300) == 1.0
    assert compute_length_damping(5000) == 1.0

    # Negative and zero hit floor
    assert compute_length_damping(0) == 0.40
    assert compute_length_damping(-50) == 0.40

    # Very short hits floor
    # (35 / 250)**0.5 = 0.374 -> clamped to 0.40
    assert compute_length_damping(35) == 0.40

    # Intermediate values
    d100 = compute_length_damping(100)
    assert pytest.approx(d100, abs=1e-4) == (100 / 250) ** 0.5
    assert 0.40 < d100 < 1.0

    d200 = compute_length_damping(200)
    assert pytest.approx(d200, abs=1e-4) == (200 / 250) ** 0.5
    assert d100 < d200 < 1.0

    # Custom threshold
    assert compute_length_damping(100, min_words=100) == 1.0
    assert pytest.approx(compute_length_damping(50, min_words=100), abs=1e-4) == 0.5**0.5


def test_explain_aligning_traits():
    text_a = (
        "The scientific treatise was meticulously constructed; furthermore, each experiment "
        "demonstrated high fidelity; however, subsequent trials yielded divergent conclusions."
    )
    text_b = (
        "The historical analysis was methodically developed; consequently, each chapter "
        "exhibited rigorous argumentation; nevertheless, further investigations revealed nuances."
    )
    doc_a = ingest(text_a)
    doc_b = ingest(text_b)
    fp_a = create_fingerprint(doc_a, label="AuthorA")
    fp_b = create_fingerprint(doc_b, label="AuthorB")

    traits = explain_aligning_traits(fp_a, fp_b, top_n=4)
    assert isinstance(traits, list)
    assert len(traits) > 0
    assert len(traits) <= 4

    for t in traits:
        assert "feature" in t
        assert "name" in t
        assert "description" in t
        assert "z_delta" in t
        assert "candidate_val" in t
        assert "essay_val" in t
        assert t["z_delta"] >= 0

    summary = format_aligning_traits_summary(traits)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_cli_identify_short_vs_long_document(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "test_exp.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    # Enroll author with substantial text
    author_sample = tmp_path / "author_profile.txt"
    sample_text = (
        "In contemporary distributed systems architecture, consistency invariants dictate "
        "protocol behavior. Nodes coordinate state machine updates across unreliable "
        "asynchronous network boundaries. Consensus protocols like Raft and Multi-Paxos "
        "establish linearizable event logs through leader election and majority quorum leases. "
        "When network partitions divide cluster communication channels, unreachable minor "
        "partitions must suspend state progression to avert catastrophic data divergence. "
        "Furthermore, durability guarantees necessitate sequential log commits prior to client "
        "acknowledgment; however, asynchronous batching pipelines mitigate excessive disk "
        "synchronization latencies. Engineers therefore navigate delicate trade-offs "
        "balancing throughput against failover convergence times. "
    ) * 4  # > 250 words
    author_sample.write_text(sample_text)
    runner.invoke(app, ["enroll", "DistSystems", str(author_sample)])

    # Short essay (< 250 words)
    short_essay = tmp_path / "short_essay.txt"
    short_essay.write_text(
        "In contemporary distributed systems, consensus quorums maintain state replication. "
        "Raft clusters select leaders through majority leases; however, network cuts halt progress."
    )

    # Long essay (> 250 words)
    long_essay = tmp_path / "long_essay.txt"
    long_essay.write_text(sample_text)

    # Test Short Essay in Table mode: should show alert banner and damping info
    res_short_tbl = runner.invoke(app, ["identify", str(short_essay), "--no-report"])
    assert res_short_tbl.exit_code == 0
    assert "Short Document Notice" in res_short_tbl.output
    assert "damped from" in res_short_tbl.output
    assert "Top Aligning Linguistic Traits" in res_short_tbl.output

    # Test Short Essay in JSON mode
    res_short_json = runner.invoke(
        app, ["identify", str(short_essay), "--format", "json", "--no-report"]
    )
    assert res_short_json.exit_code == 0
    short_data = json.loads(res_short_json.output)
    assert short_data["short_document"] is True
    assert short_data["length_warning"] == SHORT_DOC_WARNING
    assert short_data["confidence"] < short_data["raw_confidence"]
    assert len(short_data["aligning_traits"]) > 0

    # Test Short Essay in CSV mode
    res_short_csv = runner.invoke(
        app, ["identify", str(short_essay), "--format", "csv", "--no-report"]
    )
    assert res_short_csv.exit_code == 0
    csv_rows = list(csv.reader(res_short_csv.output.strip().splitlines()))
    assert "top_aligning_traits" in csv_rows[0]
    assert len(csv_rows[1][6]) > 0  # top_aligning_traits string present

    # Test Long Essay in Table mode: should NOT show alert banner
    res_long_tbl = runner.invoke(app, ["identify", str(long_essay), "--no-report"])
    assert res_long_tbl.exit_code == 0
    assert "Short Document Notice" not in res_long_tbl.output
    assert "damped from" not in res_long_tbl.output
    assert "Top Aligning Linguistic Traits" in res_long_tbl.output

    # Test Long Essay in JSON mode
    res_long_json = runner.invoke(
        app, ["identify", str(long_essay), "--format", "json", "--no-report"]
    )
    assert res_long_json.exit_code == 0
    long_data = json.loads(res_long_json.output)
    assert long_data["short_document"] is False
    assert long_data["length_warning"] is None
    assert long_data["confidence"] == long_data["raw_confidence"]


def test_cli_short_doc_analyze_and_compare(tmp_path: Path):
    short_file1 = tmp_path / "short1.txt"
    short_file1.write_text("A brief note with very few words.")
    short_file2 = tmp_path / "short2.txt"
    short_file2.write_text("Another small note with minimal word length.")

    # Analyze single file JSON
    res_a_json = runner.invoke(
        app, ["analyze", str(short_file1), "--format", "json", "--no-report"]
    )
    assert res_a_json.exit_code == 0
    a_data = json.loads(res_a_json.output)
    assert a_data["short_document"] is True
    assert a_data["length_warning"] == SHORT_DOC_WARNING

    # Analyze single file Table
    res_a_tbl = runner.invoke(app, ["analyze", str(short_file1), "--no-report"])
    assert res_a_tbl.exit_code == 0
    assert "Short Document Notice" in res_a_tbl.output

    # Compare JSON
    res_c_json = runner.invoke(
        app,
        [
            "compare",
            str(short_file1),
            str(short_file2),
            "--format",
            "json",
            "--no-report",
        ],
    )
    assert res_c_json.exit_code == 0
    c_data = json.loads(res_c_json.output)
    assert c_data["short_document"] is True
    assert c_data["length_warning"] == SHORT_DOC_WARNING
    assert "file1_words" in c_data
    assert "file2_words" in c_data

    # Compare Table
    res_c_tbl = runner.invoke(app, ["compare", str(short_file1), str(short_file2), "--no-report"])
    assert res_c_tbl.exit_code == 0
    assert "Short Document Notice" in res_c_tbl.output
