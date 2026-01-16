"""Evaluate SandwichSP model on test data with per-residue MCC metrics."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.utils import resample
from tqdm import tqdm

# Add src to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sandwichsp import SandwichSP
from sandwichsp.config import Config

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Constants
SEQ_LENGTH = 70
LABEL_MAP = Config.LABEL_MAP
LABEL_MAP_INV = Config.LABEL_MAP_INV

# 6-to-4 State Mapping
SIX_TO_FOUR_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}
PLOT_LABELS_4STATE = ['SP', 'TAT', 'LIPO', 'OTHER']
PLOT_LABELS_6STATE = ['SP', 'TAT', 'LIPO', 'NO_SP', 'NO_SP_B', 'OTHER']


def load_test_data(csv_path: str) -> tuple[list[str], list[list[int]]]:
    """Load test data and convert labels to integers."""
    df = pd.read_csv(csv_path)

    # Filter out sequences with 'P' (pilin) labels
    df = df[~df["labels"].str.contains("P", na=False)]

    sequences = df["sequence"].tolist()

    # Convert string labels to integer sequences
    label_seqs = []
    for label_str in df["labels"]:
        label_seq = [LABEL_MAP[c] for c in label_str if c in LABEL_MAP]
        label_seqs.append(label_seq)

    return sequences, label_seqs


def predict_all(model: SandwichSP, sequences: list[str], batch_size: int = 8) -> list[list[int]]:
    """Run predictions on all sequences."""
    pred_seqs = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
        batch = sequences[i:i + batch_size]
        results = model.predict_batch(batch)

        for result in results:
            # Convert label string to integers
            pred_seq = [LABEL_MAP[c] for c in result.labels]
            pred_seqs.append(pred_seq)

    return pred_seqs


def map_to_4state(sequences: list[list[int]]) -> list[list[int]]:
    """Map 6-state sequences to 4-state."""
    return [[SIX_TO_FOUR_MAP.get(label, 3) for label in seq] for seq in sequences]


def position_specific_metrics(pred_seqs: list[list[int]], true_seqs: list[list[int]],
                               seq_length: int = 70) -> tuple[np.ndarray, np.ndarray]:
    """Calculate MCC and Accuracy at each position."""
    mccs = []
    accs = []

    for i in range(seq_length):
        true_at_pos = []
        pred_at_pos = []

        for t_seq, p_seq in zip(true_seqs, pred_seqs):
            if i < len(t_seq) and i < len(p_seq):
                true_at_pos.append(t_seq[i])
                pred_at_pos.append(p_seq[i])

        if len(true_at_pos) > 0:
            if len(set(true_at_pos).union(set(pred_at_pos))) > 1:
                mccs.append(matthews_corrcoef(true_at_pos, pred_at_pos))
            else:
                mccs.append(0.0)
            accs.append(accuracy_score(true_at_pos, pred_at_pos))
        else:
            mccs.append(0.0)
            accs.append(0.0)

    return np.array(mccs), np.array(accs)


def bootstrap_stats(pred_seqs: list[list[int]], true_seqs: list[list[int]],
                    n_iterations: int = 1000, seq_length: int = 70) -> tuple:
    """Calculate bootstrap statistics for MCC and Accuracy."""
    print(f"Running bootstrap analysis ({n_iterations} iterations)...")

    bootstrap_mccs = []
    bootstrap_accs = []

    n_samples = len(pred_seqs)
    indices = np.arange(n_samples)

    for _ in tqdm(range(n_iterations), desc="Bootstrap"):
        resampled_idx = resample(indices, replace=True)
        p_resampled = [pred_seqs[i] for i in resampled_idx]
        t_resampled = [true_seqs[i] for i in resampled_idx]

        mccs, accs = position_specific_metrics(p_resampled, t_resampled, seq_length)
        bootstrap_mccs.append(mccs)
        bootstrap_accs.append(accs)

    mccs_arr = np.array(bootstrap_mccs)
    accs_arr = np.array(bootstrap_accs)

    return (np.mean(mccs_arr, axis=0), np.std(mccs_arr, axis=0),
            np.mean(accs_arr, axis=0), np.std(accs_arr, axis=0))


def plot_metrics(mean_mcc, std_mcc, mean_acc, std_acc, output_path: Path, title: str):
    """Plot MCC and Accuracy with error bands."""
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(1, len(mean_mcc) + 1)

    ax.plot(x, mean_mcc, label='MCC', color='#1f77b4', linewidth=2)
    ax.fill_between(x, mean_mcc - std_mcc, mean_mcc + std_mcc,
                    color='#1f77b4', alpha=0.2, label='MCC Std Dev')

    ax.plot(x, mean_acc, label='Accuracy', color='#ff7f0e', linewidth=2, linestyle='--')
    ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc,
                    color='#ff7f0e', alpha=0.2, label='Accuracy Std Dev')

    ax.set_xlabel('AA Position', fontsize=12)
    ax.set_ylabel('Metric Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='-', alpha=0.6)
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_label_distribution(sequences: list[list[int]], output_path: Path,
                            title: str, num_states: int = 6):
    """Plot label distribution across positions."""
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
        labels = PLOT_LABELS_4STATE
    else:
        colors = ['#D55E00', '#009E73', '#56B4E9', '#E69F00', '#0072B2', '#CCCCCC']
        labels = PLOT_LABELS_6STATE

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, SEQ_LENGTH + 1)

    ax.stackplot(x, *ratios, labels=labels, colors=colors, alpha=0.9)
    ax.set_xlim(1, SEQ_LENGTH)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('AA Position', fontsize=12)
    ax.set_ylabel('Relative Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='center right', framealpha=0.95, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Distribution plot saved to: {output_path}")
    plt.close()


def compute_global_metrics(pred_seqs: list[list[int]], true_seqs: list[list[int]]) -> dict:
    """Compute global metrics across all positions."""
    all_preds = []
    all_trues = []

    for p_seq, t_seq in zip(pred_seqs, true_seqs):
        min_len = min(len(p_seq), len(t_seq))
        all_preds.extend(p_seq[:min_len])
        all_trues.extend(t_seq[:min_len])

    return {
        'accuracy': accuracy_score(all_trues, all_preds),
        'mcc': matthews_corrcoef(all_trues, all_preds),
    }


def evaluate(csv_path: str, output_dir: str = None, device: str = "auto",
             use_6state: bool = True, n_bootstrap: int = 1000):
    """Run full evaluation pipeline."""

    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = Path(__file__).parent / "evaluation_results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_suffix = "6state" if use_6state else "4state"
    num_states = 6 if use_6state else 4

    print(f"Loading test data from {csv_path}...")
    sequences, true_seqs = load_test_data(csv_path)
    print(f"Loaded {len(sequences)} sequences")

    # Filter to sequences with at least SEQ_LENGTH positions
    filtered_data = [(s, t) for s, t in zip(sequences, true_seqs) if len(t) >= SEQ_LENGTH]
    sequences = [s for s, _ in filtered_data]
    true_seqs = [t[:SEQ_LENGTH] for _, t in filtered_data]
    print(f"After filtering (>= {SEQ_LENGTH} AA): {len(sequences)} sequences")

    print(f"\nInitializing SandwichSP (device={device})...")
    model = SandwichSP(device=device)

    print("\nRunning predictions...")
    pred_seqs = predict_all(model, sequences)
    pred_seqs = [p[:SEQ_LENGTH] for p in pred_seqs]

    # Map to 4-state if needed
    if not use_6state:
        print("Mapping to 4-state...")
        pred_seqs = map_to_4state(pred_seqs)
        true_seqs = map_to_4state(true_seqs)

    # Global metrics
    print("\nComputing global metrics...")
    global_metrics = compute_global_metrics(pred_seqs, true_seqs)
    print(f"  Global Accuracy: {global_metrics['accuracy']:.4f}")
    print(f"  Global MCC: {global_metrics['mcc']:.4f}")

    # Bootstrap analysis
    mean_mcc, std_mcc, mean_acc, std_acc = bootstrap_stats(
        pred_seqs, true_seqs, n_iterations=n_bootstrap, seq_length=SEQ_LENGTH
    )

    # Save metrics to CSV
    df = pd.DataFrame({
        'Position': np.arange(1, SEQ_LENGTH + 1),
        'Mean_MCC': mean_mcc,
        'Std_MCC': std_mcc,
        'Mean_Accuracy': mean_acc,
        'Std_Accuracy': std_acc,
    })
    csv_out = output_dir / f"sandwichsp_metrics_{state_suffix}.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nMetrics saved to: {csv_out}")

    # Plot metrics
    plot_metrics(
        mean_mcc, std_mcc, mean_acc, std_acc,
        output_dir / f"sandwichsp_metrics_{state_suffix}.png",
        f"SandwichSP Performance ({num_states}-state): MCC vs Accuracy"
    )

    # Plot distributions
    plot_label_distribution(
        true_seqs,
        output_dir / f"ground_truth_distribution_{state_suffix}.png",
        f"Ground Truth Label Distribution ({num_states}-state)",
        num_states=num_states
    )

    plot_label_distribution(
        pred_seqs,
        output_dir / f"predicted_distribution_{state_suffix}.png",
        f"Predicted Label Distribution ({num_states}-state)",
        num_states=num_states
    )

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Test sequences: {len(sequences)}")
    print(f"Evaluation mode: {num_states}-state")
    print(f"Global Accuracy: {global_metrics['accuracy']:.4f}")
    print(f"Global MCC: {global_metrics['mcc']:.4f}")
    print(f"Mean position MCC: {mean_mcc.mean():.4f} (+/- {std_mcc.mean():.4f})")
    print(f"Mean position Accuracy: {mean_acc.mean():.4f} (+/- {std_acc.mean():.4f})")
    print(f"\nResults saved to: {output_dir}")

    return {
        'global_metrics': global_metrics,
        'position_metrics': df,
        'mean_mcc': mean_mcc,
        'std_mcc': std_mcc,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SandwichSP model")
    parser.add_argument("--csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--use-6state", action="store_true", help="Use 6-state (default: 6-state)")
    parser.add_argument("--use-4state", action="store_true", help="Use 4-state evaluation")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")

    args = parser.parse_args()

    use_6state = not args.use_4state

    evaluate(
        csv_path=args.csv,
        output_dir=args.output,
        device=args.device,
        use_6state=use_6state,
        n_bootstrap=args.bootstrap,
    )
