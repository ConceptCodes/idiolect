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
    import json

    data = json.loads(result.output)
    assert data["label"] == "River_Sample"
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


def test_cli_identify_no_candidates(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "empty.db"
    import idiolect.store

    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    essay = tmp_path / "essay.txt"
    essay.write_text("An anonymous paper on thermodynamics and molecular kinetics.")
    res = runner.invoke(app, ["identify", str(essay)])
    assert res.exit_code != 0
    assert "No students or authors enrolled" in res.output


def test_cli_identify_success(tmp_path: Path, monkeypatch):
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

    # Test identify with PDF report
    res_id = runner.invoke(app, ["identify", str(unknown_essay)])
    assert res_id.exit_code == 0
    assert "AUTHOR IDENTIFICATION" in res_id.output
    assert "Top Match: Alice" in res_id.output
    assert "Candidate Ranking" in res_id.output
    assert (tmp_path / "artifacts" / "identify_submission_Alice.pdf").exists()

    # Test identify with --no-report
    res_no_rep = runner.invoke(app, ["identify", str(unknown_essay), "--no-report"])
    assert res_no_rep.exit_code == 0
    assert "Top Match: Alice" in res_no_rep.output


def test_cli_analyze_batch_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.txt").write_text(
        "The rapid development of modern artificial intelligence tools and models."
    )
    (corpus / "doc2.txt").write_text(
        "I walked slowly through the dense pines listening to the chirping birds."
    )

    # With reports
    out_dir = tmp_path / "custom_artifacts"
    res = runner.invoke(app, ["analyze", str(corpus), "--output", str(out_dir)])
    assert res.exit_code == 0
    assert "Batch Linguistic Analysis" in res.output
    assert "BATCH ANALYSIS COMPLETE" in res.output
    assert (out_dir / "doc1_fingerprint.pdf").exists()
    assert (out_dir / "doc2_fingerprint.pdf").exists()

    # Empty directory error
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    res_empty = runner.invoke(app, ["analyze", str(empty_dir)])
    assert res_empty.exit_code != 0
    assert "No supported text files found" in res_empty.output


def test_cli_enroll_and_verify_batch(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "batch_test.db"
    import idiolect.store

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    # 1. Enroll from folder of writing samples
    author_samples = tmp_path / "clara_samples"
    author_samples.mkdir()
    (author_samples / "sample1.txt").write_text(
        "Her meticulous attention to detail was evident across every stanza of poetry."
    )
    (author_samples / "sample2.txt").write_text(
        "The lyrical phrasing carried an unmistakable cadence that enchanted all listeners."
    )

    res_enroll = runner.invoke(app, ["enroll", "Clara", str(author_samples)])
    assert res_enroll.exit_code == 0
    assert "Successfully enrolled author profile: Clara" in res_enroll.output
    assert "2 samples" in res_enroll.output

    # 2. Verify folder of documents against Clara
    test_folder = tmp_path / "verify_folder"
    test_folder.mkdir()
    (test_folder / "test1.txt").write_text(
        "Her lyrical phrasing and poetic stanzas left a lasting impression on everyone."
    )
    (test_folder / "test2.txt").write_text(
        "The compiler optimizes register allocation using graph coloring heuristics."
    )

    res_verify = runner.invoke(app, ["verify", "Clara", str(test_folder)])
    assert res_verify.exit_code == 0
    assert "Batch Author Verification Against 'Clara'" in res_verify.output
    assert "BATCH VERIFICATION COMPLETE" in res_verify.output


def test_cli_identify_batch_directory(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "students_batch.db"
    import idiolect.store

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    # Enroll Candidate A (Emily) and Candidate B (David)
    cand_dir = tmp_path / "candidates"
    cand_dir.mkdir()
    f_emily = cand_dir / "emily.txt"
    f_david = cand_dir / "david.txt"
    f_emily.write_text(
        "In the quiet laboratory, Eleanor observed the glowing crystals beneath the lens. "
        "Her notes detailed the delicate luminescence shimmering across the glass."
    )
    f_david.write_text(
        "Distributed database replication protocols must ensure serializable isolation levels. "
        "Network partitioning triggers consensus quorums under Paxos invariants."
    )

    runner.invoke(app, ["enroll", "Emily", str(f_emily)])
    runner.invoke(app, ["enroll", "David", str(f_david)])

    # Submissions folder with 2 essays
    subs_dir = tmp_path / "submissions"
    subs_dir.mkdir()
    (subs_dir / "sub1.txt").write_text(
        "Eleanor continued examining the illuminated glass prisms in the silent laboratory, "
        "recording each refractive measurement carefully in her leather notebook."
    )
    (subs_dir / "sub2.txt").write_text(
        "Consensus voting during cluster network splits requires a majority quorum. "
        "Replicated state machine logs preserve linearizable consistency."
    )

    # Run batch identify
    res_batch = runner.invoke(app, ["identify", str(subs_dir)])
    assert res_batch.exit_code == 0
    assert "Batch Author Identification" in res_batch.output
    assert "BATCH IDENTIFICATION COMPLETE" in res_batch.output
    assert "Emily" in res_batch.output
    assert "David" in res_batch.output
    assert (tmp_path / "artifacts" / "identify_sub1_Emily.pdf").exists()
    assert (tmp_path / "artifacts" / "identify_sub2_David.pdf").exists()

    # Run batch identify with --no-report
    res_no_rep = runner.invoke(app, ["identify", str(subs_dir), "--no-report"])
    assert res_no_rep.exit_code == 0
    assert "Batch Author Identification" in res_no_rep.output


def test_cli_multi_sample_enroll_and_profile(tmp_path: Path, monkeypatch):
    test_db = tmp_path / "multi_sample_cli.db"
    import idiolect.store

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    s1 = tmp_path / "essay1.txt"
    s2 = tmp_path / "essay2.txt"
    s1.write_text(
        "The crisp morning frost covered the quiet garden stones as autumn set in gently."
    )
    s2.write_text(
        "Winter approached swiftly, bringing dark twilight and freezing evening gusts outside."
    )

    # 1. First sample
    res1 = runner.invoke(app, ["enroll", "Arthur", str(s1)])
    assert res1.exit_code == 0
    assert "Sample #1: essay1.txt" in res1.output

    # 2. Second sample (rolling update)
    res2 = runner.invoke(app, ["enroll", "Arthur", str(s2)])
    assert res2.exit_code == 0
    assert "Added sample to author profile: Arthur" in res2.output
    assert "Updated Rolling Baseline: 2 samples" in res2.output

    # 3. List profiles
    res_list = runner.invoke(app, ["list"])
    assert res_list.exit_code == 0
    assert "Arthur" in res_list.output
    assert "Enrolled Author Profiles" in res_list.output

    # 4. Profile inspection
    res_prof = runner.invoke(app, ["profile", "Arthur"])
    assert res_prof.exit_code == 0
    assert "IDIOLECT AUTHOR PROFILE" in res_prof.output
    assert "Arthur" in res_prof.output
    assert "Enrolled Writing Samples" in res_prof.output
    assert "essay1.txt" in res_prof.output
    assert "essay2.txt" in res_prof.output
    assert "Rolling Composite Baseline" in res_prof.output

    # 5. Profile inspection with PDF output
    out_dir = tmp_path / "prof_artifacts"
    res_pdf = runner.invoke(app, ["profile", "Arthur", "--output", str(out_dir)])
    assert res_pdf.exit_code == 0
    assert (out_dir / "profile_Arthur.pdf").exists()

    # 6. Profile non-existent author
    res_missing = runner.invoke(app, ["profile", "NonExistent"])
    assert res_missing.exit_code != 0
    assert "not enrolled" in res_missing.output


def test_cli_analyze_formats(tmp_path: Path):
    import csv
    import json

    f1 = tmp_path / "doc1.txt"
    f2 = tmp_path / "doc2.txt"
    f1.write_text("The silent forest shrouded the forgotten path under ancient oak trees.")
    f2.write_text("Rapid rivers cut through the narrow granite canyons over countless centuries.")

    # 1. Single file JSON
    res_json = runner.invoke(app, ["analyze", str(f1), "--format", "json", "--no-report"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["word_count"] > 0
    assert "axes" in data
    assert "standout_traits" in data

    # 2. Single file CSV
    res_csv = runner.invoke(app, ["analyze", str(f1), "--format", "csv", "--no-report"])
    assert res_csv.exit_code == 0
    lines = list(csv.reader(res_csv.output.strip().splitlines()))
    assert len(lines) == 2
    assert "document" in lines[0]
    assert lines[1][0] == "doc1.txt"

    # 3. Batch Directory JSON
    res_b_json = runner.invoke(app, ["analyze", str(tmp_path), "--format", "json", "--no-report"])
    assert res_b_json.exit_code == 0
    b_data = json.loads(res_b_json.output)
    assert isinstance(b_data, list)
    assert len(b_data) == 2
    doc_names = {item["document"] for item in b_data}
    assert doc_names == {"doc1.txt", "doc2.txt"}

    # 4. Batch Directory CSV
    res_b_csv = runner.invoke(app, ["analyze", str(tmp_path), "--format", "csv", "--no-report"])
    assert res_b_csv.exit_code == 0
    b_lines = list(csv.reader(res_b_csv.output.strip().splitlines()))
    assert len(b_lines) == 3  # header + 2 docs
    assert b_lines[0][0] == "document"
    assert {b_lines[1][0], b_lines[2][0]} == {"doc1.txt", "doc2.txt"}

    # 5. Invalid format
    res_inv = runner.invoke(app, ["analyze", str(f1), "--format", "xml"])
    assert res_inv.exit_code != 0
    assert "Invalid format 'xml'" in res_inv.output


def test_cli_identify_formats(tmp_path: Path, monkeypatch):
    import csv
    import json

    import idiolect.store

    test_db = tmp_path / "identify_formats.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    # Enroll authors
    s_emily = tmp_path / "emily_train.txt"
    s_emily.write_text(
        "In the quiet laboratory, Eleanor observed the glowing crystals beneath the lens. "
        "Her notes detailed the delicate luminescence shimmering across the glass."
    )
    runner.invoke(app, ["enroll", "Emily", str(s_emily)])

    s_david = tmp_path / "david_train.txt"
    s_david.write_text(
        "Distributed database replication protocols must ensure serializable isolation levels. "
        "Network partitioning triggers consensus quorums under Paxos invariants."
    )
    runner.invoke(app, ["enroll", "David", str(s_david)])

    # Submissions
    subs_dir = tmp_path / "submissions"
    subs_dir.mkdir()
    sub1 = subs_dir / "essay_emily.txt"
    sub1.write_text(
        "Eleanor continued examining the illuminated glass prisms in the silent laboratory, "
        "recording each refractive measurement carefully in her leather notebook."
    )
    sub2 = subs_dir / "essay_david.txt"
    sub2.write_text(
        "Consensus voting during cluster network splits requires a majority quorum. "
        "Replicated state machine logs preserve linearizable consistency."
    )

    # 1. Single file JSON
    res_json = runner.invoke(app, ["identify", str(sub1), "--format", "json", "--no-report"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["essay"] == "essay_emily.txt"
    assert data["top_match"] == "Emily"
    assert len(data["candidates"]) == 2
    assert data["short_document"] is True
    assert "aligning_traits" in data

    # 2. Single file CSV
    res_csv = runner.invoke(app, ["identify", str(sub1), "--format", "csv", "--no-report"])
    assert res_csv.exit_code == 0
    lines = list(csv.reader(res_csv.output.strip().splitlines()))
    assert lines[0] == [
        "rank",
        "candidate",
        "samples",
        "confidence",
        "burrows_delta",
        "verdict",
        "top_aligning_traits",
    ]
    assert lines[1][1] == "Emily"

    # 3. Batch Directory CSV (LMS mode)
    res_b_csv = runner.invoke(app, ["identify", str(subs_dir), "--format", "csv", "--no-report"])
    assert res_b_csv.exit_code == 0
    b_lines = list(csv.reader(res_b_csv.output.strip().splitlines()))
    assert b_lines[0] == [
        "submission",
        "words",
        "short_doc",
        "top_match",
        "confidence",
        "burrows_delta",
        "verdict",
        "lead_margin",
        "top_aligning_traits",
    ]
    assert len(b_lines) == 3
    submissions_matched = {row[0]: row[3] for row in b_lines[1:]}
    assert submissions_matched["essay_emily.txt"] == "Emily"
    assert submissions_matched["essay_david.txt"] == "David"

    # 4. Batch Directory JSON
    res_b_json = runner.invoke(app, ["identify", str(subs_dir), "--format", "json", "--no-report"])
    assert res_b_json.exit_code == 0
    b_data = json.loads(res_b_json.output)
    assert len(b_data) == 2
    b_map = {item["submission"]: item["top_match"] for item in b_data}
    assert b_map["essay_emily.txt"] == "Emily"
    assert b_map["essay_david.txt"] == "David"
    assert b_data[0]["short_document"] is True
    assert "top_aligning_traits" in b_data[0]


def test_cli_verify_and_compare_formats(tmp_path: Path, monkeypatch):
    import csv
    import json

    import idiolect.store

    test_db = tmp_path / "verify_formats.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    sample = tmp_path / "sample.txt"
    sample.write_text("Philosophical treatises explore epistemic modalities and ontological truth.")
    runner.invoke(app, ["enroll", "Socrates", str(sample)])

    # Verify single JSON (short text damped)
    res_v_json = runner.invoke(app, ["verify", "Socrates", str(sample), "--format", "json"])
    assert res_v_json.exit_code == 0
    v_data = json.loads(res_v_json.output)
    assert v_data["author"] == "Socrates"
    assert v_data["short_document"] is True
    assert v_data["raw_confidence"] > 90
    assert v_data["confidence"] <= 45  # Damped due to 8-word length

    # Verify single CSV
    res_v_csv = runner.invoke(app, ["verify", "Socrates", str(sample), "--format", "csv"])
    assert res_v_csv.exit_code == 0
    v_lines = list(csv.reader(res_v_csv.output.strip().splitlines()))
    assert v_lines[0] == [
        "document",
        "author",
        "words",
        "short_doc",
        "confidence",
        "burrows_delta",
        "verdict",
    ]
    assert v_lines[1][1] == "Socrates"
    assert v_lines[1][3] == "yes"

    # Compare JSON & CSV
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("The moon climbed over misty peaks as the night grew chill.")
    f2.write_text("The silver moon crested the foggy mountaintops in the cold air.")

    res_c_json = runner.invoke(
        app, ["compare", str(f1), str(f2), "--format", "json", "--no-report"]
    )
    assert res_c_json.exit_code == 0
    c_data = json.loads(res_c_json.output)
    assert "similarity" in c_data
    assert "axis_deltas" in c_data

    res_c_csv = runner.invoke(app, ["compare", str(f1), str(f2), "--format", "csv", "--no-report"])
    assert res_c_csv.exit_code == 0
    c_lines = list(csv.reader(res_c_csv.output.strip().splitlines()))
    assert c_lines[0] == ["axis", "delta"]


def test_cli_list_and_profile_formats(tmp_path: Path, monkeypatch):
    import csv
    import json

    import idiolect.store

    test_db = tmp_path / "list_prof.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "idiolect.cli.get_store",
        lambda db_path=None: idiolect.store.FingerprintStore(db_path=test_db),
    )

    # Empty list
    res_empty_json = runner.invoke(app, ["list", "--format", "json"])
    assert res_empty_json.exit_code == 0
    assert json.loads(res_empty_json.output) == []

    res_empty_csv = runner.invoke(app, ["list", "--format", "csv"])
    assert res_empty_csv.exit_code == 0
    assert "author,samples,words,author_type,consistency,top_trait" in res_empty_csv.output

    # Enroll
    s1 = tmp_path / "sample1.txt"
    s1.write_text("The ancient parchment held secrets written in cryptographic ciphers.")
    runner.invoke(app, ["enroll", "Scholar", str(s1)])

    # List JSON
    res_list_json = runner.invoke(app, ["list", "--format", "json"])
    assert res_list_json.exit_code == 0
    list_data = json.loads(res_list_json.output)
    assert len(list_data) == 1
    assert list_data[0]["author"] == "Scholar"

    # List CSV
    res_list_csv = runner.invoke(app, ["list", "--format", "csv"])
    assert res_list_csv.exit_code == 0
    csv_rows = list(csv.reader(res_list_csv.output.strip().splitlines()))
    assert csv_rows[1][0] == "Scholar"

    # Profile JSON
    res_prof_json = runner.invoke(app, ["profile", "Scholar", "--format", "json"])
    assert res_prof_json.exit_code == 0
    prof_data = json.loads(res_prof_json.output)
    assert prof_data["author"] == "Scholar"
    assert prof_data["samples_count"] == 1
    assert len(prof_data["samples"]) == 1

    # Profile CSV
    res_prof_csv = runner.invoke(app, ["profile", "Scholar", "--format", "csv"])
    assert res_prof_csv.exit_code == 0
    prof_rows = list(csv.reader(res_prof_csv.output.strip().splitlines()))
    assert prof_rows[0] == [
        "sample_index",
        "sample_label",
        "words",
        "enrolled_at",
        "rolling_weight",
    ]
    assert prof_rows[1][1] == "sample1.txt"
