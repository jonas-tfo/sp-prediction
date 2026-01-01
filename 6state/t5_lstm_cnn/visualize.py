"""Visualization functions for training metrics and model architecture."""

import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .config import Config


def plot_training_metrics(results_path: str = None):
    """
    Plot training metrics from saved results.

    Args:
        results_path: Path to pickle file with training results.
    """
    results_path = results_path or (Config.OUTPUT_DIR / "6state_t5_lstm_cnn_fold_results.pkl")

    with open(results_path, 'rb') as f:
        fold_results = pickle.load(f)

    num_folds = fold_results['num_folds']

    # Plot validation metrics across folds
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Colors for each fold
    colors = plt.cm.tab10(np.linspace(0, 1, num_folds))

    # Plot 1: Training and Validation Loss
    ax = axes[0, 0]
    for i, (train_loss, val_loss) in enumerate(zip(fold_results['train_losses'], fold_results['val_losses'])):
        epochs_range = range(1, len(train_loss) + 1)
        ax.plot(epochs_range, train_loss, linestyle='--', color=colors[i], alpha=0.5, label=f'Fold {i+1} Train')
        ax.plot(epochs_range, val_loss, linestyle='-', color=colors[i], label=f'Fold {i+1} Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Token-level Accuracy
    ax = axes[0, 1]
    for i, token_accs in enumerate(fold_results['val_token_acc']):
        epochs_range = range(1, len(token_accs) + 1)
        ax.plot(epochs_range, token_accs, color=colors[i], marker='o', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Token Accuracy')
    ax.set_title('Validation Token-level Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Sequence-level Accuracy
    ax = axes[0, 2]
    for i, seq_accs in enumerate(fold_results['val_seq_acc']):
        epochs_range = range(1, len(seq_accs) + 1)
        ax.plot(epochs_range, seq_accs, color=colors[i], marker='s', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sequence Accuracy')
    ax.set_title('Validation Sequence-level Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: MCC
    ax = axes[1, 0]
    for i, mccs in enumerate(fold_results['val_mcc']):
        epochs_range = range(1, len(mccs) + 1)
        ax.plot(epochs_range, mccs, color=colors[i], marker='^', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MCC')
    ax.set_title('Validation Matthews Correlation Coefficient')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Precision
    ax = axes[1, 1]
    for i, precs in enumerate(fold_results['val_precision']):
        epochs_range = range(1, len(precs) + 1)
        ax.plot(epochs_range, precs, color=colors[i], marker='d', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Validation Precision (weighted)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: Recall
    ax = axes[1, 2]
    for i, recs in enumerate(fold_results['val_recall']):
        epochs_range = range(1, len(recs) + 1)
        ax.plot(epochs_range, recs, color=colors[i], marker='v', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Validation Recall (weighted)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('K-Fold Cross Validation Metrics (T5 LSTM-CNN)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = Config.OUTPUT_DIR / '6state_t5_lstm_cnn_metrics_plot.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Metrics plot saved to: {plot_path}")
    plt.show()

    return fig


def plot_best_metrics_bar(results_path: str = None):
    """
    Plot bar chart of best metrics across folds.

    Args:
        results_path: Path to pickle file with training results.
    """
    results_path = results_path or (Config.OUTPUT_DIR / "6state_t5_lstm_cnn_fold_results.pkl")

    with open(results_path, 'rb') as f:
        fold_results = pickle.load(f)

    num_folds = fold_results['num_folds']
    all_fold_metrics = fold_results['best_metrics']

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['Token Acc', 'Seq Acc', 'MCC', 'Precision', 'Recall']
    x = np.arange(len(metrics_names))
    width = 0.5

    all_values = {
        'token_acc': [m['token_acc'] for m in all_fold_metrics],
        'seq_acc': [m['seq_acc'] for m in all_fold_metrics],
        'mcc': [m['mcc'] for m in all_fold_metrics],
        'precision': [m['precision'] for m in all_fold_metrics],
        'recall': [m['recall'] for m in all_fold_metrics]
    }

    # Calculate mean values
    mean_values = [
        np.mean(all_values['token_acc']),
        np.mean(all_values['seq_acc']),
        np.mean(all_values['mcc']),
        np.mean(all_values['precision']),
        np.mean(all_values['recall'])
    ]

    if len(all_fold_metrics) > 1:
        min_values = [
            min(all_values['token_acc']),
            min(all_values['seq_acc']),
            min(all_values['mcc']),
            min(all_values['precision']),
            min(all_values['recall'])
        ]

        max_values = [
            max(all_values['token_acc']),
            max(all_values['seq_acc']),
            max(all_values['mcc']),
            max(all_values['precision']),
            max(all_values['recall'])
        ]

        # Calculate asymmetric error bars
        yerr_lower = [max(0, mean - min_val) for mean, min_val in zip(mean_values, min_values)]
        yerr_upper = [max(0, max_val - mean) for max_val, mean in zip(max_values, mean_values)]
    else:
        yerr_lower = [0] * len(mean_values)
        yerr_upper = [0] * len(mean_values)

    # Plot bars
    ax.bar(x, mean_values, width, color='steelblue', edgecolor='none', zorder=2)
    ax.errorbar(x, mean_values, yerr=[yerr_lower, yerr_upper], fmt='none',
                ecolor='black', capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)

    # Styling
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')

    # Horizontal lines
    yticks = np.arange(0, 1.1, 0.2)
    for y_tick in yticks:
        ax.axhline(y=y_tick, color='lightgray', linewidth=0.8, zorder=3)

    ax.set_ylabel('Score')
    ax.set_title(f'Mean Validation Metrics Across {num_folds} Folds (Error bars: min/max)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(yticks)

    # Value labels
    max_yerr = max(yerr_upper) if max(yerr_upper) > 0 else 0.02
    for i, val in enumerate(mean_values):
        ax.text(i, val + max_yerr + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    bar_path = Config.OUTPUT_DIR / '6state_t5_lstm_cnn_best_metrics_bar.png'
    plt.savefig(bar_path, dpi=150)
    print(f"Bar plot saved to: {bar_path}")
    plt.show()

    return fig


def draw_model_architecture():
    """Draw a vertical diagram of the model architecture."""
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.axis('off')

    # Define blocks with (label, x, y)
    blocks = [
        ("Input Embeddings\n(Pre-computed T5)", 1.5, 9.0),
        ("Attention Mask", 1.5, 8.2),
        ("Conv1D\n(1024 → 1024, kernel=5)", 1.5, 7.0),
        ("BatchNorm1D + ReLU", 1.5, 6.0),
        ("BiLSTM\n(2 layers, 1024 → 512×2)", 1.5, 5.0),
        ("Classifier\n(Linear: 1024 → 6)", 1.5, 4.0),
        ("Dropout\n(p=0.35)", 1.5, 3.0),
        ("CRF Layer\n(6 states)", 1.5, 2.0),
        ("Per Token Predictions", 1.5, 1.0)
    ]

    box_width = 2.0
    box_height = 0.6

    # Draw blocks
    for label, x, y in blocks:
        rect = mpatches.FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.03",
            edgecolor='black', facecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + box_width / 2, y + box_height / 2, label, ha='center', va='center', fontsize=10)

    # Draw arrows between consecutive blocks
    for i in range(len(blocks) - 1):
        x1 = blocks[i][1] + box_width / 2
        y1 = blocks[i][2]
        y2 = blocks[i+1][2] + box_height
        ax.annotate('', xy=(x1, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.title("SPCNNClassifier Architecture (T5)", fontsize=14, fontweight='bold')
    plt.ylim(0, 10)
    plt.xlim(0, 5)
    plt.tight_layout()

    arch_path = Config.OUTPUT_DIR / 'model_architecture.png'
    plt.savefig(arch_path, dpi=150)
    print(f"Architecture diagram saved to: {arch_path}")
    plt.show()

    return fig


def plot_all(results_path: str = None):
    """Generate all plots."""
    plot_training_metrics(results_path)
    plot_best_metrics_bar(results_path)
    draw_model_architecture()


if __name__ == "__main__":
    plot_all()
