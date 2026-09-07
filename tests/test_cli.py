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


def test_cli_analyze_default_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = tmp_path / "stream.txt"
    sample.write_text("A crystal mountain stream tumbled over smooth granite rocks in the valley.")
    result = runner.invoke(app, ["analyze", str(sample)])
    assert result.exit_code == 0
    assert "Report saved" in result.output
    assert (tmp_path / "artifacts" / "stream_fingerprint.pdf").exists()


def test_cli_compare_default_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f1 = tmp_path / "docA.txt"
    f2 = tmp_path / "docB.txt"
    f1.write_text("Morning sunlight filtered through the pine needles onto the forest floor.")
    f2.write_text("Dawn beams pierced through evergreen boughs above the mossy ground.")
    result = runner.invoke(app, ["compare", str(f1), str(f2)])
    assert result.exit_code == 0
    assert "Comparison report saved" in result.output
    assert (tmp_path / "artifacts" / "compare_docA_docB.pdf").exists()


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


def test_cli_guess_no_candidates(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "empty.db"
    import idiolect.store

    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    essay = tmp_path / "essay.txt"
    essay.write_text("An anonymous paper on thermodynamics and molecular kinetics.")
    res = runner.invoke(app, ["guess", str(essay)])
    assert res.exit_code != 0
    assert "No students or authors enrolled" in res.output


def test_cli_guess_and_identify_success(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "students.db"
    import idiolect.store

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    # Enroll Student A (Alice) and Student B (Bob)
    sample_a = tmp_path / "alice.txt"
    sample_b = tmp_path / "bob.txt"
    sample_a.write_text(
        "In our tranquil laboratory, the morning fog hung low over the glassware. "
        "Eleanor hurried past the rows of test tubes, clutching an unread manuscript."
    )
    sample_b.write_text(
        "Computational systems demonstrate profound variance across distributed cluster "
        "environments. Latency profiles degrade proportionally as concurrency exceeds limits."
    )

    runner.invoke(app, ["enroll", "Alice", str(sample_a)])
    runner.invoke(app, ["enroll", "Bob", str(sample_b)])

    # Mystery essay written in Alice's literary style
    unknown_essay = tmp_path / "submission.txt"
    unknown_essay.write_text(
        "Through the quiet village streets, Eleanor wandered among the shuttered workshops, "
        "wondering if the whispers from the northern harbor could possibly be true."
    )

    # Test guess
    res_guess = runner.invoke(app, ["guess", str(unknown_essay)])
    assert res_guess.exit_code == 0
    assert "AUTHOR IDENTIFICATION / GUESS" in res_guess.output
    assert "Top Guess: Alice" in res_guess.output
    assert "Candidate Ranking" in res_guess.output
    assert (tmp_path / "artifacts" / "guess_submission_Alice.pdf").exists()

    # Test identify alias with --no-report
    res_id = runner.invoke(app, ["identify", str(unknown_essay), "--no-report"])
    assert res_id.exit_code == 0
    assert "Top Guess: Alice" in res_id.output
