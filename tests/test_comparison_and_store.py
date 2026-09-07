from idiolect.comparison import compare
from idiolect.fingerprint import create_fingerprint_from_text
from idiolect.store import FingerprintStore

def test_comparison():
    text1 = "I love hiking in the woods and watching the sunlight filter through the tall trees."
    text2 = "I enjoy walking in the forest and seeing the rays of sunlight through the pines."
    
    fp1 = create_fingerprint_from_text(text1, label="sample1")
    fp2 = create_fingerprint_from_text(text2, label="sample2")
    
    res = compare(fp1, fp2)
    assert 0.0 <= res.cosine_similarity <= 1.0
    assert res.same_author_likelihood in ["very_likely", "likely", "uncertain", "unlikely", "very_unlikely"]
    assert len(res.axis_deltas) == 7

def test_store(tmp_path):
    db_file = tmp_path / "test_store.db"
    store = FingerprintStore(db_path=db_file)
    
    text = "This is an enrolled writing sample for test author Jane Doe."
    fp = create_fingerprint_from_text(text, label="Jane Doe")
    
    store.enroll(fp)
    assert "Jane Doe" in store.list_all()
    
    retrieved = store.get("Jane Doe")
    assert retrieved is not None
    assert retrieved.label == "Jane Doe"
    
    assert store.delete("Jane Doe") is True
    assert "Jane Doe" not in store.list_all()
