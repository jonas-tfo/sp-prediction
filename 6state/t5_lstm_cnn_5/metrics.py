
from typing import List
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score


def sequence_level_accuracy(preds_flat: List[int], labels_flat: List[int], label_seqs: List[List[int]]) -> float:

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


# TODO actually use this in the training, currently being done within train file
def compute_metrics(all_preds: List[int], all_labels: List[int], label_seqs: List[List[int]] = None) -> dict:

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
