from idiolect.fingerprint import create_fingerprint_from_text
from idiolect.models import AuthorType, Fingerprint


def test_fingerprint_creation_and_serialization():
    text = (
        "I was walking through the forest when I stumbled upon an old, forgotten cabin. "
        "The roof had collapsed, and moss covered the porch. "
        "Could anyone have lived here recently? I certainly doubted it."
    )
    fp = create_fingerprint_from_text(text, label="test_doc")
    assert isinstance(fp, Fingerprint)
    assert fp.label == "test_doc"
    assert len(fp.axes) == 7
    assert 0 <= fp.axes["lexical_richness"] <= 100
    assert 0 <= fp.axes["formality"] <= 100
    assert fp.author_type in [AuthorType.HUMAN, AuthorType.AI, AuthorType.UNCERTAIN]

    # Test JSON round-trip
    json_str = fp.to_json()
    fp_restored = Fingerprint.from_json(json_str)
    assert fp_restored.label == fp.label
    assert fp_restored.author_type == fp.author_type
    assert fp_restored.axes == fp.axes
