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
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm  # Optional: for progress bar during bootstrapping

from .config import Config
from .dataset import SPDatasetWithEmbeddings
from .metrics import sequence_level_accuracy, sequence_level_accuracy_only_sps
from .model import SPCNNClassifier
from .utils import get_test_data

def position_specific_mcc(pred_sequences, true_sequences):
    """
    Calculates MCC per position (0 to 69).
    Requires sequences to be of fixed length 70.
    """
    mccs = []
    # Ensure we don't go out of bounds if sequences are shorter (though logic ensures len=70)
    seq_len = len(true_sequences[0]) if true_sequences else 70
    
    for i in range(seq_len):
        true_flat = [seq[i] for seq in true_sequences]
        pred_flat = [seq[i] for seq in pred_sequences]
        mccs.append(matthews_corrcoef(true_flat, pred_flat))

    return mccs

def calculate_bootstrap_std(pred_seqs, label_seqs, n_iterations=1000):
    """
    Performs bootstrap resampling to calculate the standard deviation 
    for MCC, Seq Acc, and Seq Acc (SPs).
    """
    print(f"\nRunning bootstrap analysis ({n_iterations} iterations)...")
    
    mcc_scores = []
    seq_acc_scores = []
    seq_acc_sps_scores = []
    
    # We iterate n times, resampling the test set with replacement
    for _ in range(n_iterations):
        # Resample at the sequence level to maintain sequence integrity
        p_resampled, l_resampled = resample(pred_seqs, label_seqs, replace=True)
        
        # Flatten predictions/labels for token-level MCC
        # Note: We must flatten the new resampled lists
        flat_preds = [token for seq in p_resampled for token in seq]
        flat_labels = [token for seq in l_resampled for token in seq]
        
        # 1. MCC (Token level)
        mcc = matthews_corrcoef(flat_labels, flat_preds)
        mcc_scores.append(mcc)
        
        # 2. Sequence Accuracy
        # Pass resampled lists to metric function
        s_acc = sequence_level_accuracy(flat_preds, flat_labels, l_resampled)
        seq_acc_scores.append(s_acc)
        
        # 3. Sequence Accuracy (Only SPs)
        s_acc_sps = sequence_level_accuracy_only_sps(flat_preds, flat_labels, l_resampled)
        seq_acc_sps_scores.append(s_acc_sps)
        
    return {
        'mcc_std': np.std(mcc_scores),
        'seq_acc_std': np.std(seq_acc_scores),
        'seq_acc_only_sps_std': np.std(seq_acc_sps_scores)
    }

def evaluate_model(embeddings_path):

    print(f"Loading model from {Config.MODEL_SAVE_PATH}")
    print(f"Using device: {Config.DEVICE}")

    # Load test data
    test_seqs, test_label_seqs, test_df = get_test_data()
    print(f"Test sequences: {len(test_seqs)}")

    # Create test dataset and loader
    test_dataset = SPDatasetWithEmbeddings(
        Config.TEST_CSV,
        embeddings_path,
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load model
    model = SPCNNClassifier(
        embedding_dim=Config.EMBEDDING_DIM,
        num_labels=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
        lstm_hidden=Config.LSTM_HIDDEN,
        lstm_layers=Config.LSTM_LAYERS,
        conv_filters=Config.CONV_FILTERS,
    ).to(Config.DEVICE)

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
    print(f"Model loaded from {Config.MODEL_SAVE_PATH}")

    # Evaluate
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    
    # For position-specific MCC (fixed length only)
    fixed_len_pred_sequences = []
    fixed_len_label_sequences = []
    
    # For Bootstrapping (ALL sequences)
    full_pred_sequences = []
    full_label_sequences = []

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
                seq_preds = []
                seq_labels = []
                for pred, true, is_valid in zip(pred_seq, label_seq, mask):
                    if true.item() != -100 and is_valid.item() == 1:
                        all_preds.append(pred)
                        all_labels.append(true.item())
                        seq_preds.append(pred)
                        seq_labels.append(true.item())
                
                # Store ALL sequences for Global Metrics & Bootstrapping
                full_pred_sequences.append(seq_preds)
                full_label_sequences.append(seq_labels)

                # Store length-70 sequences for Position-Specific MCC
                if len(seq_preds) == 70: 
                    fixed_len_pred_sequences.append(seq_preds)
                    fixed_len_label_sequences.append(seq_labels)

    # Calculate Standard Metrics
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
    
    # Sequence level metrics (Point estimates)
    seq_acc = sequence_level_accuracy(all_preds, all_labels, full_label_sequences)
    seq_acc_only_sps = sequence_level_accuracy_only_sps(all_preds, all_labels, full_label_sequences)
    avg_loss = test_loss / len(test_loader)

    # ---------------------------------------------------------
    # Bootstrap Analysis for Std Dev
    # ---------------------------------------------------------
    bootstrap_results = calculate_bootstrap_std(full_pred_sequences, full_label_sequences)
    mcc_std = bootstrap_results['mcc_std']
    seq_acc_std = bootstrap_results['seq_acc_std']
    seq_acc_only_sps_std = bootstrap_results['seq_acc_only_sps_std']

    print(f"\nMetrics Summary:")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Token-level Accuracy: {token_acc:.4f}")
    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("-" * 30)
    print(f"MCC: {mcc:.4f} ± {mcc_std:.4f}")
    print(f"Sequence Level Accuracy: {seq_acc:.4f} ± {seq_acc_std:.4f}")
    print(f"Sequence Level Accuracy (only SPs): {seq_acc_only_sps:.4f} ± {seq_acc_only_sps_std:.4f}")
    print("-" * 30)

    # Position-specific MCC
    pos_mccs = position_specific_mcc(fixed_len_pred_sequences, fixed_len_label_sequences)
    print(f"\nPosition-specific MCC ({len(fixed_len_pred_sequences)} sequences, len=70):")
    print(f"  Mean: {np.mean(pos_mccs):.4f}")
    print(f"  Std: {np.std(pos_mccs):.4f}")
    print(f"  Min: {np.min(pos_mccs):.4f} (position {np.argmin(pos_mccs)})")
    print(f"  Max: {np.max(pos_mccs):.4f} (position {np.argmax(pos_mccs)})")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(Config.LABEL_MAP.values()))
    cm_relative = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_relative, display_labels=list(Config.LABEL_MAP.keys()))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix - Final Model")
    plt.tight_layout()

    # Save confusion matrix
    cm_path = Config.PLOTS_SAVE_DIR / "confusion_matrix_final.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to: {cm_path}")
    # plt.show() # Optional: Comment out if running on server

    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'mcc': mcc,
        'mcc_std': mcc_std,
        'token_acc': token_acc,
        'seq_acc': seq_acc,
        'seq_acc_std': seq_acc_std,
        'seq_acc_only_sps': seq_acc_only_sps,
        'seq_acc_only_sps_std': seq_acc_only_sps_std,
        'avg_loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'position_specific_mcc': pos_mccs,
    }

if __name__ == "__main__":
    res = evaluate_model(Config.TEST_EMBEDINGS)
    metrics_path = Config.PLOTS_SAVE_DIR / "evaluation_metrics_final.txt"
    with open(metrics_path, "w") as f:
        for key, value in res.items():
            if isinstance(value, list):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
