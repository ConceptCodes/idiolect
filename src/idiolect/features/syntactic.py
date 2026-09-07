"""Syntactic feature extraction for idiolect.

Extracts features related to sentence structure, POS distributions,
and dependency parsing patterns.
"""

import numpy as np

from ..models import Document


def extract_syntactic(doc: Document) -> dict[str, float]:
    """Extract syntactic features from a Document."""
    features = {
        "sent_length_mean": 0.0,
        "sent_length_std": 0.0,
        "sent_length_cv": 0.0,
        "sent_length_min": 0.0,
        "sent_length_max": 0.0,
        "parse_tree_depth_mean": 0.0,
        "parse_tree_depth_max": 0.0,
        "dep_arc_length_mean": 0.0,
        "passive_voice_ratio": 0.0,
        "subordinate_clause_ratio": 0.0,
    }

    # Universal POS tags
    pos_tags = [
        "NOUN",
        "VERB",
        "ADJ",
        "ADV",
        "PRON",
        "DET",
        "ADP",
        "AUX",
        "CCONJ",
        "SCONJ",
        "PUNCT",
        "NUM",
        "PROPN",
        "PART",
        "INTJ",
    ]
    for tag in pos_tags:
        features[f"pos_{tag}"] = 0.0

    if not doc.tokens or not doc.sentences:
        return features

    spacy_doc = doc.spacy_doc

    # Sentence lengths (in tokens, not just words)
    sent_lengths = [len(list(sent)) for sent in spacy_doc.sents if len(sent.text.strip()) > 0]
    if sent_lengths:
        mean_len = float(np.mean(sent_lengths))
        std_len = float(np.std(sent_lengths))
        features["sent_length_mean"] = mean_len
        features["sent_length_std"] = std_len
        features["sent_length_cv"] = std_len / mean_len if mean_len > 0 else 0.0
        features["sent_length_min"] = float(np.min(sent_lengths))
        features["sent_length_max"] = float(np.max(sent_lengths))

    # Dependency features
    def get_tree_depth(token):
        children = list(token.children)
        if not children:
            return 0
        return 1 + max(get_tree_depth(child) for child in children)

    tree_depths = []
    arc_lengths = []
    passive_count = 0
    subordinate_count = 0
    total_deps = 0
    pos_counts = {tag: 0 for tag in pos_tags}
    total_pos = 0

    for sent in spacy_doc.sents:
        if not sent.text.strip():
            continue
        tree_depths.append(get_tree_depth(sent.root))

        for token in sent:
            # POS
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
            total_pos += 1

            # Dependency arc length
            arc_lengths.append(abs(token.head.i - token.i))
            total_deps += 1

            # Passive and subordinates
            if token.dep_ in ("auxpass", "nsubjpass"):
                passive_count += 1
            if token.dep_ in ("advcl", "relcl", "ccomp", "xcomp"):
                subordinate_count += 1

    if tree_depths:
        features["parse_tree_depth_mean"] = float(np.mean(tree_depths))
        features["parse_tree_depth_max"] = float(np.max(tree_depths))

    if arc_lengths:
        features["dep_arc_length_mean"] = float(np.mean(arc_lengths))

    if total_deps > 0:
        features["passive_voice_ratio"] = passive_count / total_deps
        features["subordinate_clause_ratio"] = subordinate_count / total_deps

    if total_pos > 0:
        for tag in pos_tags:
            features[f"pos_{tag}"] = pos_counts[tag] / total_pos

    return features
