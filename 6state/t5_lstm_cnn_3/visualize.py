
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .config import Config


def plot_training_metrics(results_path: Path  = Config.TRAIN_VAL_LOSSES_PKL_SAVE_PATH):

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

    plot_path = Config.PLOTS_SAVE_DIR / '6state_t5_lstm_cnn_2_metrics_plot.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Metrics plot saved to: {plot_path}")

    return fig


def plot_best_metrics_bar(results_path: Path = Config.TRAIN_VAL_LOSSES_PKL_SAVE_PATH):

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

    # Calculate standard deviation for error bars
    if len(all_fold_metrics) > 1:
        std_values = [
            np.std(all_values['token_acc']),
            np.std(all_values['seq_acc']),
            np.std(all_values['mcc']),
            np.std(all_values['precision']),
            np.std(all_values['recall'])
        ]
    else:
        std_values = [0] * len(mean_values)

    # Plot bars
    ax.bar(x, mean_values, width, color='steelblue', edgecolor='none', zorder=2)
    ax.errorbar(x, mean_values, yerr=std_values, fmt='none',
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
    ax.set_title(f'Mean Validation Metrics Across {num_folds} Folds (Error bars: ±1 std)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(yticks)

    # Value labels with mean ± std
    max_yerr = max(std_values) if max(std_values) > 0 else 0.02
    for i, (val, std) in enumerate(zip(mean_values, std_values)):
        ax.text(i, val + max_yerr + 0.02, f'{val:.3f} ± {std:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    bar_path = Config.PLOTS_SAVE_DIR / '6state_t5_lstm_cnn_2_best_metrics_bar.png'
    plt.savefig(bar_path, dpi=150)
    print(f"Bar plot saved to: {bar_path}")

    return fig


def draw_model_architecture():
    """Draw a vertical diagram of the model architecture based on SPCNNClassifier in model.py."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')

    box_width = 2.0
    box_height = 0.6

    # Top row: Input Embeddings and Attention Mask side by side
    top_blocks = [
        ("Input Embeddings\n(Pre-computed T5, dim=1024)", 0.5, 9.0),
        ("Attention Mask", 4.5, 9.0),
    ]

    # Main architecture blocks (centered)
    main_blocks = [
        ("Conv1D\n(1024 → 1024, kernel=5, pad=2)", 2.5, 7.5),
        ("BatchNorm1D + ReLU", 2.5, 6.5),
        ("BiLSTM\n(3 layers, 1024 → 512×2)", 2.5, 5.5),
        ("Classifier\n(Linear: 1024 → 6)", 2.5, 4.5),
        ("Dropout\n(p=0.35)", 2.5, 3.5),
        ("CRF Layer\n(6 states)", 2.5, 2.5),
        ("Per Token Predictions", 2.5, 1.5)
    ]

    # Draw top blocks
    for label, x, y in top_blocks:
        rect = mpatches.FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.03",
            edgecolor='black', facecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + box_width / 2, y + box_height / 2, label, ha='center', va='center', fontsize=10)

    # Draw main blocks
    for label, x, y in main_blocks:
        rect = mpatches.FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.03",
            edgecolor='black', facecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + box_width / 2, y + box_height / 2, label, ha='center', va='center', fontsize=10)

    # Draw arrows from both top blocks to first main block
    first_main = main_blocks[0]
    for top_block in top_blocks:
        x_start = top_block[1] + box_width / 2
        y_start = top_block[2]
        x_end = first_main[1] + box_width / 2
        y_end = first_main[2] + box_height
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Draw arrows between consecutive main blocks
    for i in range(len(main_blocks) - 1):
        x1 = main_blocks[i][1] + box_width / 2
        y1 = main_blocks[i][2]
        y2 = main_blocks[i+1][2] + box_height
        ax.annotate('', xy=(x1, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.title("SPCNNClassifier Architecture (T5)", fontsize=14, fontweight='bold')
    plt.ylim(0, 10)
    plt.xlim(0, 7)
    plt.tight_layout()

    arch_path = Config.PLOTS_SAVE_DIR / 'model_architecture_6state_t5_lstm_cnn_2.png'
    plt.savefig(arch_path, dpi=150)
    print(f"Architecture diagram saved to: {arch_path}")

    return fig


def plot_all(results_path: Path = Config.TRAIN_VAL_LOSSES_PKL_SAVE_PATH):

    plot_training_metrics(results_path)
    plot_best_metrics_bar(results_path)
    draw_model_architecture()


if __name__ == "__main__":
    plot_all()



