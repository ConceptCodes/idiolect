from pathlib import Path

import pytest

from idiolect.fingerprint import create_fingerprint
from idiolect.ingestion import ingest
from idiolect.profiling import (
    aggregate_fingerprints,
    calculate_sample_weights,
    classify_stability,
    compute_profile_consistency,
)
from idiolect.store import FingerprintStore


def test_calculate_sample_weights_single():
    assert calculate_sample_weights([1000]) == [1.0]
    assert calculate_sample_weights([]) == []


def test_calculate_sample_weights_multiple():
    # 3 samples with equal word count: newest should have highest weight due to recency decay
    weights = calculate_sample_weights([1000, 1000, 1000], recency_decay=0.90)
    assert len(weights) == 3
    assert pytest.approx(sum(weights), abs=1e-5) == 1.0
    assert weights[2] > weights[1] > weights[0]

    # Large sample earlier vs small sample later
    # Sample 1: 5000 words (age=1, 5000 * 0.9 = 4500)
    # Sample 2: 500 words (age=0, 500 * 1.0 = 500)
    w = calculate_sample_weights([5000, 500], recency_decay=0.90)
    assert w[0] > w[1]  # Volume outweighs recency here
    assert pytest.approx(sum(w), abs=1e-5) == 1.0


def test_calculate_sample_weights_max_samples():
    # 5 samples, max_samples=3
    weights = calculate_sample_weights([100, 200, 300, 400, 500], max_samples=3)
    assert len(weights) == 3
    assert pytest.approx(sum(weights), abs=1e-5) == 1.0


def test_aggregate_fingerprints_single():
    doc = ingest("The brilliant sunlight warmed the green leaves on the high branches.")
    fp = create_fingerprint(doc, label="sample1")
    composite = aggregate_fingerprints([fp], label="Alice")

    assert composite.label == "Alice"
    assert composite.sample_count == 1
    assert composite.word_count == fp.word_count
    assert composite.axes == fp.axes
    assert composite.features == fp.features


def test_aggregate_fingerprints_multi_and_stability():
    doc1 = ingest("A solitary crow perched upon the ancient oak tree in the quiet courtyard.")
    doc2 = ingest(
        "Modern statistical methodologies enable systematic stylometric classification of texts."
    )
    doc3 = ingest(
        "Through the winding forest trail, the traveler continued walking under moonlight."
    )

    fp1 = create_fingerprint(doc1, label="s1")
    fp2 = create_fingerprint(doc2, label="s2")
    fp3 = create_fingerprint(doc3, label="s3")

    composite = aggregate_fingerprints([fp1, fp2, fp3], label="CompositeAuthor", recency_decay=0.90)

    assert composite.label == "CompositeAuthor"
    assert composite.sample_count == 3
    assert composite.word_count == fp1.word_count + fp2.word_count + fp3.word_count
    assert composite.sentence_count == (
        fp1.sentence_count + fp2.sentence_count + fp3.sentence_count
    )

    # Stability should have positive standard deviation across axes
    assert len(composite.axis_stability) == 7
    for ax, std in composite.axis_stability.items():
        assert std >= 0.0

    consistency = compute_profile_consistency(composite.axis_stability)
    assert 0.0 <= consistency <= 100.0


def test_classify_stability():
    assert classify_stability(1.5) == "Highly Stable"
    assert classify_stability(4.2) == "Stable"
    assert classify_stability(8.0) == "Moderate Var."
    assert classify_stability(15.0) == "High Var."


def test_store_multi_sample_lifecycle(tmp_path: Path):
    db_path = tmp_path / "profiles.db"
    store = FingerprintStore(db_path=db_path)

    doc1 = ingest("The morning light cascaded across the quiet water in the valley.")
    doc2 = ingest("Eleanor examined the delicate crystal specimens with scholarly interest.")
    doc3 = ingest("Deep within the ancient archives, forgotten manuscripts lay undisturbed.")

    fp1 = create_fingerprint(doc1, label="essay1")
    fp2 = create_fingerprint(doc2, label="essay2")
    fp3 = create_fingerprint(doc3, label="essay3")

    # 1. Enroll first sample
    c1 = store.enroll_sample("Alice", fp1, sample_label="essay1.txt")
    assert c1.sample_count == 1
    assert c1.word_count == fp1.word_count

    # Check store.get
    alice_fp = store.get("Alice")
    assert alice_fp is not None
    assert alice_fp.sample_count == 1

    # 2. Enroll second sample (updates rolling baseline)
    c2 = store.enroll_sample("Alice", fp2, sample_label="essay2.txt")
    assert c2.sample_count == 2
    assert c2.word_count == fp1.word_count + fp2.word_count

    # 3. Enroll third sample
    c3 = store.enroll_sample("Alice", fp3, sample_label="essay3.txt")
    assert c3.sample_count == 3
    assert c3.word_count == fp1.word_count + fp2.word_count + fp3.word_count

    # 4. Inspect samples and profile
    samples = store.get_samples("Alice")
    assert len(samples) == 3
    assert samples[0].sample_label == "essay1.txt"
    assert samples[1].sample_label == "essay2.txt"
    assert samples[2].sample_label == "essay3.txt"

    profile = store.get_profile("Alice")
    assert profile is not None
    assert profile.sample_count == 3
    assert profile.total_word_count == c3.word_count

    # 5. Delete a specific sample (e.g. sample 2)
    sample_to_del = samples[1].id
    assert sample_to_del is not None
    updated_comp = store.delete_sample("Alice", sample_to_del)
    assert updated_comp is not None
    assert updated_comp.sample_count == 2
    assert updated_comp.word_count == fp1.word_count + fp3.word_count

    remaining_samples = store.get_samples("Alice")
    assert len(remaining_samples) == 2
    assert [s.sample_label for s in remaining_samples] == ["essay1.txt", "essay3.txt"]

    # 6. Replace enrollment
    doc4 = ingest("A completely new baseline for Alice.")
    fp4 = create_fingerprint(doc4, label="new_baseline")
    replaced = store.enroll_sample("Alice", fp4, sample_label="fresh.txt", replace=True)
    assert replaced.sample_count == 1
    assert len(store.get_samples("Alice")) == 1
    assert store.get_samples("Alice")[0].sample_label == "fresh.txt"
