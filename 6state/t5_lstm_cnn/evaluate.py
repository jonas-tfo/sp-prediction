"""Evaluation for test set."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt

from .config import Config
from .dataset import SPDatasetWithEmbeddings
from .metrics import sequence_level_accuracy
from .model import SPCNNClassifier
from .utils import get_test_data


def evaluate_model(model_path: str = None, embeddings_path: str = None):

    model_path = model_path or Config.MODEL_SAVE_PATH
    embeddings_path = embeddings_path or (Config.OUTPUT_DIR / "test_embeddings.npz")

    print(f"Loading model from {model_path}")
    print(f"Using device: {Config.DEVICE}")

    # Load test data
    test_seqs, test_label_seqs, test_df = get_test_data()
    print(f"Test sequences: {len(test_seqs)}")

    # Create test dataset and loader
    test_dataset = SPDatasetWithEmbeddings(
        Config.TEST_CSV,
        embeddings_path,
        Config.LABEL_MAP,
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load model
    model = SPCNNClassifier(
        embedding_dim=Config.EMBEDDING_DIM,
        num_labels=Config.NUM_CLASSES,
    ).to(Config.DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    print(f"Model loaded from {model_path}")

    # Evaluate
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            # Compute loss using CRF
            loss = model(embeddings, attention_mask, labels)
            test_loss += loss.item()

            # Decode predictions using CRF
            predictions = model(embeddings, attention_mask)

            # Collect valid tokens (skip -100 labels)
            for pred_seq, label_seq, mask in zip(predictions, labels, attention_mask):
                for pred, true, is_valid in zip(pred_seq, label_seq, mask):
                    if true.item() != -100 and is_valid.item() == 1:
                        all_preds.append(pred)
                        all_labels.append(true.item())

    # Calculate metrics
    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(Config.LABEL_MAP.keys())))

    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    mcc = matthews_corrcoef(all_labels, all_preds)
    token_acc = accuracy_score(all_labels, all_preds)
    seq_acc = sequence_level_accuracy(all_preds, all_labels, test_label_seqs)
    avg_loss = test_loss / len(test_loader)

    print(f"\nMetrics Summary:")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Token-level Accuracy: {token_acc:.4f}")
    print(f"Sequence Level Accuracy: {seq_acc:.4f}")
    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(Config.LABEL_MAP.values()))
    cm_relative = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_relative, display_labels=list(Config.LABEL_MAP.keys()))
    disp.plot(cmap="OrRd", xticks_rotation=45)
    plt.title("Confusion Matrix - T5 LSTM-CNN")
    plt.tight_layout()

    # Save confusion matrix
    cm_path = Config.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to: {cm_path}")
    plt.show()

    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'mcc': mcc,
        'token_acc': token_acc,
        'seq_acc': seq_acc,
        'avg_loss': avg_loss,
        'precision': precision,
        'recall': recall,
    }


if __name__ == "__main__":
    evaluate_model()
