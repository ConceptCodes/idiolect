from idiolect.features.character import extract_character
from idiolect.features.lexical import extract_lexical
from idiolect.features.pragmatic import extract_pragmatic
from idiolect.features.readability import extract_readability
from idiolect.features.syntactic import extract_syntactic
from idiolect.ingestion import ingest


def test_feature_extractors():
    sample = (
        "Although we were completely exhausted after the long hiking trip, "
        "I couldn't help but marvel at the majestic mountain peak! "
        "Why didn't we bring our camera? It was truly unforgettable."
    )
    doc = ingest(sample)

    lex = extract_lexical(doc)
    assert "mattr" in lex
    assert "yule_k" in lex
    assert lex["avg_word_length"] > 0

    syn = extract_syntactic(doc)
    assert "sent_length_mean" in syn
    assert "sent_length_cv" in syn
    assert "parse_tree_depth_mean" in syn
    assert syn["pos_NOUN"] >= 0

    read = extract_readability(doc)
    assert "flesch_reading_ease" in read
    assert "flesch_kincaid_grade" in read

    prag = extract_pragmatic(doc)
    assert "pron_first_singular" in prag
    assert "contraction_ratio" in prag
    assert "sentiment_std" in prag

    char = extract_character(doc)
    assert "punct_comma" in char
    assert "punct_exclaim" in char
    assert "question_sentence_ratio" in char
