"""Character-level and structural feature extraction for idiolect.

Extracts features related to punctuation, casing, digits,
and paragraph structure.
"""

import string
import numpy as np

from ..models import Document

def extract_character(doc: Document) -> dict[str, float]:
    """Extract character-level and structural features from a Document."""
    features = {
        "punct_comma": 0.0,
        "punct_period": 0.0,
        "punct_exclaim": 0.0,
        "punct_question": 0.0,
        "punct_semicolon": 0.0,
        "punct_colon": 0.0,
        "punct_dash": 0.0,
        "punct_ellipsis": 0.0,
        "punct_paren": 0.0,
        "punct_quote": 0.0,
        "punct_to_word_ratio": 0.0,
        "uppercase_word_ratio": 0.0,
        "capitalized_word_ratio": 0.0,
        "digit_ratio": 0.0,
        "avg_paragraph_length": 0.0,
        "char_diversity": 0.0,
        "question_sentence_ratio": 0.0,
        "exclamation_sentence_ratio": 0.0,
    }

    if not doc.tokens or not doc.all_tokens:
        return features

    # Punctuation frequencies
    n_tokens = len(doc.all_tokens) # using all_tokens for punctuation ratio
    n_words = len(doc.tokens)
    per_1k = 1000.0 / n_tokens if n_tokens > 0 else 0.0

    punct_counts = {
        "punct_comma": doc.all_tokens.count(","),
        "punct_period": doc.all_tokens.count("."),
        "punct_exclaim": doc.all_tokens.count("!"),
        "punct_question": doc.all_tokens.count("?"),
        "punct_semicolon": doc.all_tokens.count(";"),
        "punct_colon": doc.all_tokens.count(":"),
        "punct_dash": doc.all_tokens.count("-") + doc.all_tokens.count("--"),
        "punct_ellipsis": doc.all_tokens.count("..."),
        "punct_paren": doc.all_tokens.count("(") + doc.all_tokens.count(")"),
        "punct_quote": doc.all_tokens.count('"') + doc.all_tokens.count("'"),
    }

    for k, v in punct_counts.items():
        features[k] = v * per_1k

    total_punct = sum(1 for t in doc.all_tokens if t in string.punctuation or t in ("...", "--"))
    features["punct_to_word_ratio"] = total_punct / n_words if n_words > 0 else 0.0

    # Word casing (ignore single letters for ALL-CAPS, except 'I')
    upper_count = 0
    cap_count = 0
    digit_count = 0

    spacy_doc = doc.spacy_doc
    
    for token in spacy_doc:
        if token.is_punct or token.is_space:
            continue
            
        t_text = token.text
        if len(t_text) > 1 and t_text.isupper():
            upper_count += 1
            
        # Capitalized but not sentence initial
        if t_text.istitle() and token.i > 0 and spacy_doc[token.i - 1].text not in (".", "!", "?"):
            cap_count += 1
            
        if any(c.isdigit() for c in t_text):
            digit_count += 1

    features["uppercase_word_ratio"] = upper_count / n_words if n_words > 0 else 0.0
    features["capitalized_word_ratio"] = cap_count / n_words if n_words > 0 else 0.0
    features["digit_ratio"] = digit_count / n_tokens if n_tokens > 0 else 0.0

    # Paragraph length
    paragraphs = [p for p in doc.cleaned_text.split("\n\n") if p.strip()]
    if paragraphs:
        # Simple heuristic for sentence count per paragraph: count punctuation
        # For more accuracy, we could run spaCy on each paragraph, but this is faster
        para_sent_counts = [max(1, len(p.split(".")) - 1) for p in paragraphs]
        features["avg_paragraph_length"] = float(np.mean(para_sent_counts))

    # Character diversity
    # Vocabulary: a-z, digits, punctuation
    text_lower = doc.cleaned_text.lower()
    valid_chars = set(string.ascii_lowercase + string.digits + string.punctuation)
    used_chars = set(c for c in text_lower if c in valid_chars)
    features["char_diversity"] = len(used_chars) / len(valid_chars) if valid_chars else 0.0

    # Sentence endings
    sentences = doc.sentences
    if sentences:
        q_sents = sum(1 for s in sentences if s.strip().endswith("?"))
        e_sents = sum(1 for s in sentences if s.strip().endswith("!"))
        features["question_sentence_ratio"] = q_sents / len(sentences)
        features["exclamation_sentence_ratio"] = e_sents / len(sentences)

    return features
