from pathlib import Path

from typer.testing import CliRunner

from idiolect.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_analyze_no_report(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "Modern artificial intelligence techniques allow computational linguists to analyze "
        "syntactic and lexical diversity across varying genres of writing with high precision."
    )

    result = runner.invoke(app, ["analyze", str(sample_file), "--no-report"])
    assert result.exit_code == 0
    assert "IDIOLECT" in result.output
    assert "Document: sample.txt" in result.output
    assert "Lexical Richness" in result.output


def test_cli_analyze_with_json_and_pdf(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "I was wandering down by the riverside when I spotted an otter splashing in the shallows. "
        "It looked at me with curiosity before diving beneath the surface."
    )
    out_dir = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(sample_file),
            "--output",
            str(out_dir),
            "--label",
            "River_Sample",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert "Report saved" in result.output
    assert (out_dir / "River_Sample_fingerprint.pdf").exists()


def test_cli_compare(tmp_path: Path):
    f1 = tmp_path / "doc1.txt"
    f2 = tmp_path / "doc2.txt"
    f1.write_text("The autumn leaves fell gently across the damp stone path in the morning.")
    f2.write_text("Crisp red and golden leaves covered the wet stone walkway as dawn arrived.")

    result = runner.invoke(app, ["compare", str(f1), str(f2), "--no-report"])
    assert result.exit_code == 0
    assert "COMPARISON" in result.output
    assert "Similarity:" in result.output
    assert "Axis Deltas" in result.output


def test_cli_store_lifecycle(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "cli_test.db"
    # Point default DB path to test_db for isolation
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )
    import idiolect.store

    sample = tmp_path / "author_sample.txt"
    sample.write_text(
        "My dear friend, you must understand that life is too brief for petty grievances. "
        "Let us walk through the orchard and enjoy the afternoon sun."
    )

    # 1. Enroll
    res_enroll = runner.invoke(app, ["enroll", "Arthur", str(sample)])
    assert res_enroll.exit_code == 0
    assert "Successfully enrolled" in res_enroll.output

    # 2. List
    res_list = runner.invoke(app, ["list"])
    assert res_list.exit_code == 0
    assert "Arthur" in res_list.output

    # 3. Verify success
    res_verify = runner.invoke(app, ["verify", "Arthur", str(sample)])
    assert res_verify.exit_code == 0
    assert "VERIFICATION RESULT" in res_verify.output

    # 4. Verify non-existent author
    res_verify_missing = runner.invoke(app, ["verify", "UnknownAuthor", str(sample)])
    assert res_verify_missing.exit_code != 0
    assert "not enrolled" in res_verify_missing.output

    # 5. Delete
    res_delete = runner.invoke(app, ["delete", "Arthur"])
    assert res_delete.exit_code == 0
    assert "Deleted enrolled author" in res_delete.output

    # 6. Delete again (should fail)
    res_delete_again = runner.invoke(app, ["delete", "Arthur"])
    assert res_delete_again.exit_code != 0
