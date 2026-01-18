import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.utils import resample
import warnings

# Try-except block for imports
try:
    from .config import Config
    from .dataset import SPDatasetWithEmbeddings
    from .model import SPCNNClassifier
except ImportError:
    from config import Config
    from dataset import SPDatasetWithEmbeddings
    from model import SPCNNClassifier

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Configuration & Mapping ---

# 6-to-4 State Mapping
SIX_TO_FOUR_MAP = {
    0: 0, # SP
    1: 1, # TAT
    2: 2, # LIPO
    3: 3, # OTHER
    4: 3, # OTHER
    5: 3  # OTHER
}

PLOT_LABELS_4STATE = ['SP', 'TAT', 'LIPO', 'OTHER']
PLOT_LABELS_6STATE = ['SP', 'TAT', 'LIPO', 'NO_SP', 'NO_SP_B', 'OTHER']
SEQ_LENGTH = 70


def map_sequences_to_4state(sequences):
    """Maps a list of integer sequences from 6-state to 4-state."""
    mapped_seqs = []
    for seq in sequences:
        mapped_seqs.append([SIX_TO_FOUR_MAP.get(label, 3) for label in seq])
    return mapped_seqs


def position_specific_metrics(pred_sequences, true_sequences, seq_length=70):
    """Calculate MCC and Accuracy at each position."""
    mccs = []
    accs = []
    
    for i in range(seq_length):
        true_at_pos = []
        pred_at_pos = []
        
        # Collect all valid labels at this position
        for t_seq, p_seq in zip(true_sequences, pred_sequences):
            if i < len(t_seq) and i < len(p_seq):
                true_at_pos.append(t_seq[i])
                pred_at_pos.append(p_seq[i])

        if len(true_at_pos) > 0:
            # MCC
            if len(set(true_at_pos).union(set(pred_at_pos))) > 1:
                mccs.append(matthews_corrcoef(true_at_pos, pred_at_pos))
            else:
                mccs.append(0.0)
            
            # Accuracy
            accs.append(accuracy_score(true_at_pos, pred_at_pos))
        else:
            mccs.append(0.0)
            accs.append(0.0)
            
    return np.array(mccs), np.array(accs)


def calculate_bootstrap_stats(pred_seqs, label_seqs, n_iterations=1000, seq_length=70):
    """Perform bootstrap analysis to calculate Mean and STD for MCC and Accuracy."""
    print(f"Running bootstrap analysis ({n_iterations} iterations) on {len(pred_seqs)} sequences...")
    
    bootstrap_mccs = []
    bootstrap_accs = []
    
    n_samples = len(pred_seqs)
    indices = np.arange(n_samples)
    
    for _ in range(n_iterations):
        # Resample indices with replacement
        resampled_indices = resample(indices, replace=True)
        p_resampled = [pred_seqs[i] for i in resampled_indices]
        l_resampled = [label_seqs[i] for i in resampled_indices]
        
        mccs, accs = position_specific_metrics(p_resampled, l_resampled, seq_length)
        bootstrap_mccs.append(mccs)
        bootstrap_accs.append(accs)
        
    mccs_arr = np.array(bootstrap_mccs)
    accs_arr = np.array(bootstrap_accs)
    
    mean_mcc = np.mean(mccs_arr, axis=0)
    std_mcc = np.std(mccs_arr, axis=0)
    
    mean_acc = np.mean(accs_arr, axis=0)
    std_acc = np.std(accs_arr, axis=0)
    
    return mean_mcc, std_mcc, mean_acc, std_acc


def plot_metrics(mean_mcc, std_mcc, mean_acc, std_acc, fold_num, output_path):
    """Plot MCC and Accuracy with error bands."""
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(1, len(mean_mcc) + 1)
    
    # MCC (Blue)
    ax.plot(x, mean_mcc, label='MCC', color='#1f77b4', linewidth=2)
    ax.fill_between(x, mean_mcc - std_mcc, mean_mcc + std_mcc, color='#1f77b4', alpha=0.2, label='MCC Std Dev')
    
    # Accuracy (Orange)
    ax.plot(x, mean_acc, label='Accuracy', color='#ff7f0e', linewidth=2, linestyle='--')
    ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, color='#ff7f0e', alpha=0.2, label='Accuracy Std Dev')
    
    ax.set_xlabel('AA Position', fontsize=12)
    ax.set_ylabel('Metric Score', fontsize=12)
    ax.set_title(f'Best Model (Fold {fold_num}) Performance: MCC vs Accuracy', fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.6)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Metrics plot saved to: {output_path}")
    plt.close()


def plot_label_distribution(sequences, output_path, title="Label Distribution", num_states=4):
    """Plots the relative distribution of labels."""
    counts = {i: np.zeros(SEQ_LENGTH) for i in range(num_states)}
    total_counts = np.zeros(SEQ_LENGTH)

    for seq in sequences:
        for i in range(min(len(seq), SEQ_LENGTH)):
            label = seq[i]
            if label in counts:
                counts[label][i] += 1
                total_counts[i] += 1

    total_counts[total_counts == 0] = 1
    ratios = [counts[i] / total_counts for i in range(num_states)]

    if num_states == 4:
        colors = ['#D55E00', '#009E73', '#56B4E9', '#CCCCCC']
        plot_labels = PLOT_LABELS_4STATE
    else:
        colors = ['#D55E00', '#009E73', '#56B4E9', '#E69F00', '#0072B2', '#CCCCCC']
        plot_labels = PLOT_LABELS_6STATE

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, SEQ_LENGTH + 1)

    ax.stackplot(x, *ratios, labels=plot_labels, colors=colors, alpha=0.9)
    
    ax.set_xlim(1, SEQ_LENGTH)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('AA Position', fontsize=12)
    ax.set_ylabel('Relative Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='center right', framealpha=0.95, fontsize=11, title="Label Type")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Distribution plot saved to: {output_path}")
    plt.close()


def get_fold_predictions(fold_num, embeddings_path=Config.TEST_EMBEDINGS):
    """Run inference for a single fold and return raw sequences."""
    
    model_path = Config.MODEL_SAVE_PATH_TEMP.format(fold_num)
    if not Path(model_path).exists():
        print(f"  Fold {fold_num}: Model file not found at {model_path}")
        return [], []

    print(f"Loading fold {fold_num} model from {model_path}...")
    
    test_dataset = SPDatasetWithEmbeddings(Config.TEST_CSV, embeddings_path)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

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

    fold_pred_seqs = []
    fold_label_seqs = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embeddings'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            predictions = model(embeddings, attention_mask)
            
            # Robustly handle Tensor (Logits) vs List output
            if isinstance(predictions, torch.Tensor):
                if predictions.dim() == 3:
                    predictions = torch.argmax(predictions, dim=2)

            for pred_seq, label_seq, mask in zip(predictions, labels, attention_mask):
                valid_preds = []
                valid_labels = []
                
                for p, t, m in zip(pred_seq, label_seq, mask):
                    if t.item() != -100 and m.item() == 1:
                        # Handle Tensor vs Int
                        p_val = p.item() if isinstance(p, torch.Tensor) else p
                        valid_preds.append(p_val)
                        valid_labels.append(t.item())

                if len(valid_preds) >= SEQ_LENGTH: 
                    fold_pred_seqs.append(valid_preds[:SEQ_LENGTH])
                    fold_label_seqs.append(valid_labels[:SEQ_LENGTH])

    return fold_pred_seqs, fold_label_seqs


def evaluate_best_fold(best_fold_num, embeddings_path=None, use_6state=False):
    """Evaluate ONLY the best fold and run bootstrap on it.

    Args:
        best_fold_num: The fold number to evaluate.
        embeddings_path: Path to the embeddings file.
        use_6state: If True, evaluate with original 6 states. If False, map to 4 states.
    """

    if embeddings_path is None:
        embeddings_path = Config.TEST_EMBEDINGS

    num_states = 6 if use_6state else 4
    state_suffix = "6state" if use_6state else "4state"

    print(f"Using device: {Config.DEVICE}")
    print(f"Test embeddings: {embeddings_path}")
    print(f"Evaluating BEST MODEL: Fold {best_fold_num}")
    print(f"Evaluation mode: {num_states}-state")

    # 1. Collect predictions ONLY for the best fold
    p_seqs, l_seqs = get_fold_predictions(best_fold_num, embeddings_path)

    if not p_seqs:
        print(f"No predictions collected for Fold {best_fold_num}. Check model path.")
        return

    print(f"\nTotal sequences collected: {len(p_seqs)}")

    # 2. Optionally map to 4-State System
    if use_6state:
        print("Using original 6-state labels...")
        eval_preds = p_seqs
        eval_labels = l_seqs
    else:
        print("Mapping 6-state predictions to 4-state...")
        eval_preds = map_sequences_to_4state(p_seqs)
        eval_labels = map_sequences_to_4state(l_seqs)

    # 3. Plot Ground Truth Distribution
    Config.PLOTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    dist_plot_path = Config.PLOTS_SAVE_DIR / f"ground_truth_distribution_{state_suffix}.png"
    plot_label_distribution(eval_labels, dist_plot_path, title="Ground Truth Label Composition", num_states=num_states)

    # 4. Bootstrap Analysis (on the single fold's predictions)
    mean_mcc, std_mcc, mean_acc, std_acc = calculate_bootstrap_stats(
        eval_preds, eval_labels, n_iterations=1000, seq_length=SEQ_LENGTH
    )

    # 5. Plot Metrics
    metrics_plot_path = Config.PLOTS_SAVE_DIR / f"fold_{best_fold_num}_metrics_{state_suffix}.png"
    plot_metrics(mean_mcc, std_mcc, mean_acc, std_acc, best_fold_num, metrics_plot_path)

    # 6. Save CSV
    positions = np.arange(1, SEQ_LENGTH + 1)

    df = pd.DataFrame({
        'Position': positions,
        'Mean_MCC': mean_mcc,
        'Std_Dev': std_mcc,
        'Position_acc': positions,
        'Mean_Accuracy': mean_acc,
        'Std_Dev_acc': std_acc
    })

    csv_path = Config.PLOTS_SAVE_DIR / f"fold_{best_fold_num}_stats_{state_suffix}_final.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed statistics saved to: {csv_path}")
    print("\nSummary (First 5 positions):")
    print(df.head())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate best fold with MCC metrics")
    parser.add_argument("--fold", type=int, default=5, help="Fold number to evaluate (default: 5)")
    parser.add_argument("--use-6state", action="store_true", help="Use original 6 states instead of mapping to 4 states")

    args = parser.parse_args()

    evaluate_best_fold(args.fold, use_6state=args.use_6state)
