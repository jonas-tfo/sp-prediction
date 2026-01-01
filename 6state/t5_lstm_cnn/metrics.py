"""Metrics for evaluating signal peptide predictions."""

from typing import List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def sequence_level_accuracy(
    preds_flat: List[int],
    labels_flat: List[int],
    label_seqs: List[List[int]],
) -> float:
    """
    Calculate sequence-level accuracy, skipping -100 (ignored) positions.

    A sequence is correct only if ALL valid positions match.

    Args:
        preds_flat: Flat list of predictions.
        labels_flat: Flat list of labels.
        label_seqs: Original label sequences to determine sequence boundaries.

    Returns:
        Sequence-level accuracy (0.0 to 1.0).
    """
    # Reconstruct sequences from flat predictions
    seq_lengths = [len(seq) for seq in label_seqs]
    preds_seq = []
    labels_seq = []
    idx = 0
    for length in seq_lengths:
        preds_seq.append(preds_flat[idx : idx + length])
        labels_seq.append(labels_flat[idx : idx + length])
        idx += length

    # Check if valid predictions match labels
    correct = 0
    for pred, label in zip(preds_seq, labels_seq):
        is_valid = [lbl != -100 for lbl in label]
        valid_preds = [p for p, valid in zip(pred, is_valid) if valid]
        valid_labels = [lbl for lbl, valid in zip(label, is_valid) if valid]
        if valid_preds == valid_labels:
            correct += 1

    total = len(seq_lengths)
    return correct / total if total > 0 else 0.0


def compute_metrics(
    all_preds: List[int],
    all_labels: List[int],
    label_seqs: List[List[int]] = None,
) -> dict:
    """
    Compute all metrics for predictions.

    Args:
        all_preds: List of predicted labels.
        all_labels: List of true labels.
        label_seqs: Optional label sequences for sequence-level accuracy.

    Returns:
        Dictionary with all computed metrics.
    """
    metrics = {
        "token_acc": accuracy_score(all_labels, all_preds),
        "mcc": matthews_corrcoef(all_labels, all_preds),
        "precision": precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
    }

    if label_seqs is not None:
        metrics["seq_acc"] = sequence_level_accuracy(all_preds, all_labels, label_seqs)

    return metrics
