
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef

from .config import Config
from .dataset import SPDatasetWithEmbeddings
from .model import SPCNNClassifier


def position_specific_mcc(pred_sequences, true_sequences):
    """Calculate MCC at each of the 70 positions."""
    mccs = []
    for i in range(70):
        true_flat = [seq[i] for seq in true_sequences]
        pred_flat = [seq[i] for seq in pred_sequences]
        mccs.append(matthews_corrcoef(true_flat, pred_flat))
    return mccs


def evaluate_fold_model(fold_num, embeddings_path):
    """Evaluate a single fold model and return position-specific MCCs."""

    model_path = Config.MODEL_SAVE_PATH_TEMP.format(fold_num)
    print(f"Loading fold {fold_num} model from {model_path}")

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

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    # Collect predictions
    all_pred_sequences = []
    all_label_sequences = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            predictions = model(embeddings, attention_mask)

            for pred_seq, label_seq, mask in zip(predictions, labels, attention_mask):
                seq_preds = []
                seq_labels = []
                for pred, true, is_valid in zip(pred_seq, label_seq, mask):
                    if true.item() != -100 and is_valid.item() == 1:
                        seq_preds.append(pred)
                        seq_labels.append(true.item())

                if len(seq_preds) == 70:
                    all_pred_sequences.append(seq_preds)
                    all_label_sequences.append(seq_labels)

    pos_mccs = position_specific_mcc(all_pred_sequences, all_label_sequences)
    print(f"  Fold {fold_num}: {len(all_pred_sequences)} sequences, mean MCC={np.mean(pos_mccs):.4f}")

    return pos_mccs


def evaluate_all_folds(embeddings_path=None):
    """Evaluate all 5 fold models and return position-specific MCCs for each."""

    if embeddings_path is None:
        embeddings_path = Config.TEST_EMBEDINGS

    print(f"Using device: {Config.DEVICE}")
    print(f"Test embeddings: {embeddings_path}")
    print(f"Evaluating {Config.NUM_FOLDS} fold models\n")

    fold_mccs = {}

    for fold in range(1, Config.NUM_FOLDS + 1):
        try:
            mccs = evaluate_fold_model(fold, embeddings_path)
            fold_mccs[fold] = mccs
        except FileNotFoundError as e:
            print(f"  Fold {fold}: Model not found - {e}")
            fold_mccs[fold] = None

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for fold, mccs in fold_mccs.items():
        if mccs is not None:
            print(f"Fold {fold}: 70 MCCs, mean={np.mean(mccs):.4f}, std={np.std(mccs):.4f}")
        else:
            print(f"Fold {fold}: Not evaluated (model not found)")

    return fold_mccs


if __name__ == "__main__":
    results = evaluate_all_folds()

    # Save results
    output_path = Config.PLOTS_SAVE_DIR / "position_specific_mcc_per_fold.npz"

    # Filter out None values for saving
    valid_results = {f"fold_{k}": np.array(v) for k, v in results.items() if v is not None}

    if valid_results:
        np.savez(output_path, **valid_results)
        print(f"\nResults saved to: {output_path}")

        # Print the MCCs for easy access
        print("\n" + "=" * 60)
        print("Position-specific MCCs per fold (list of 70 values each):")
        print("=" * 60)
        for fold, mccs in results.items():
            if mccs is not None:
                print(f"\nFold {fold}:")
                print(mccs)
    else:
        print("\nNo valid results to save. Please run training first.")
