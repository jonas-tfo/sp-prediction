
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

    # Plot validation metrics across folds (MCC, Sequence Accuracy, and SP Sequence Accuracy)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors for each fold
    colors = plt.cm.tab10(np.linspace(0, 1, num_folds))

    # Plot 1: MCC
    ax = axes[0]
    for i, mccs in enumerate(fold_results['val_mcc']):
        epochs_range = range(1, len(mccs) + 1)
        ax.plot(epochs_range, mccs, color=colors[i], marker='^', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MCC')
    ax.set_title('Validation Matthews Correlation Coefficient')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Sequence-level Accuracy
    ax = axes[1]
    for i, seq_accs in enumerate(fold_results['val_seq_acc']):
        epochs_range = range(1, len(seq_accs) + 1)
        ax.plot(epochs_range, seq_accs, color=colors[i], marker='s', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sequence Accuracy')
    ax.set_title('Validation Sequence-level Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Sequence-level Accuracy (SP sequences only)
    ax = axes[2]
    for i, seq_accs_sp in enumerate(fold_results['val_seq_accs_only_sps']):
        epochs_range = range(1, len(seq_accs_sp) + 1)
        ax.plot(epochs_range, seq_accs_sp, color=colors[i], marker='o', markersize=3, label=f'Fold {i+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sequence Accuracy (SP only)')
    ax.set_title('Validation Seq Accuracy (SP sequences only)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('K-Fold Cross Validation Metrics (T5 LSTM-CNN)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = Config.PLOTS_SAVE_DIR / '6state_t5_lstm_cnn_6_metrics_plot.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Metrics plot saved to: {plot_path}")

    return fig


def plot_best_metrics_bar(results_path: Path = Config.TRAIN_VAL_LOSSES_PKL_SAVE_PATH):

    with open(results_path, 'rb') as f:
        fold_results = pickle.load(f)

    num_folds = fold_results['num_folds']
    all_fold_metrics = fold_results['best_metrics']

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['MCC', 'Seq Acc', 'Seq Acc (SP only)']
    x = np.arange(len(metrics_names))
    width = 0.5

    all_values = {
        'mcc': [m['mcc'] for m in all_fold_metrics],
        'seq_acc': [m['seq_acc'] for m in all_fold_metrics],
        'seq_acc_only_sps': [m['seq_acc_only_sps'] for m in all_fold_metrics],
    }

    # Calculate mean values
    mean_values = [
        np.mean(all_values['mcc']),
        np.mean(all_values['seq_acc']),
        np.mean(all_values['seq_acc_only_sps']),
    ]

    # Calculate standard deviation for error bars
    if len(all_fold_metrics) > 1:
        std_values = [
            np.std(all_values['mcc']),
            np.std(all_values['seq_acc']),
            np.std(all_values['seq_acc_only_sps']),
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
    ax.set_title(f'Mean Validation Metrics Across {num_folds} Folds (Error bars: ± std)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(yticks)

    # Value labels with mean ± std
    max_yerr = max(std_values) if max(std_values) > 0 else 0.02
    for i, (val, std) in enumerate(zip(mean_values, std_values)):
        ax.text(i, val + max_yerr + 0.02, f'{val:.3f} ± {std:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    bar_path = Config.PLOTS_SAVE_DIR / '6state_t5_lstm_cnn_6_best_metrics_bar.png'
    plt.savefig(bar_path, dpi=150)
    print(f"Bar plot saved to: {bar_path}")

    return fig


def draw_model_architecture():
    """Draw a vertical diagram of the model architecture based on SPCNNClassifier in model.py."""
    fig, ax = plt.subplots(figsize=(18, 22))
    ax.axis('off')

    box_width = 4.0
    box_height = 1.0
    font_size = 18

    # Center x position for shared layers
    center_x = 7.0

    # Top row: Input Embeddings and Attention Mask side by side
    top_blocks = [
        ("Input Embeddings\n(Pre-computed T5, dim=1024)", 4.0, 19.0),
        ("Attention Mask", 10.0, 19.0),
    ]

    # Main architecture blocks before split
    pre_split_blocks = [
        ("Conv1D (conv5)\n(1024 → 256, kernel=5, pad=2)", center_x, 17.0),
        ("BatchNorm1D + GELU", center_x, 15.5),
        ("BiLSTM\n(2 layers, 256 → 512×2)", center_x, 14.0),
    ]

    # Parallel conv branches
    parallel_blocks = [
        ("Conv1D (conv7)\n(1024 → 256, k=7, p=3)", 4.0, 12.0),
        ("Conv1D (conv9)\n(1024 → 256, k=9, p=4)", 10.0, 12.0),
    ]

    # Concat block
    concat_block = ("Concatenate\n(256 + 256 = 512)", center_x, 10.0)

    # Post-concat blocks (up to classifier)
    post_concat_blocks = [
        ("BatchNorm1D + GELU", center_x, 8.5),
        ("Dropout (p=0.126)", center_x, 7.0),
        ("Classifier\n(Linear: 512 → 6 logits)", center_x, 5.5),
    ]

    # Inference path (LEFT side)
    inference_x = 3.0
    inference_blocks = [
        ("CRF Decode\n(Viterbi)", inference_x, 3.0),
        ("Per Token\nPredictions", inference_x, 1.3),
    ]

    # Training path (RIGHT side)
    training_x = 11.0
    loss_blocks = [
        ("CRF Loss\n(neg. log-likelihood)", 8.5, 3.0),
        ("Weighted CE Loss\n(class-balanced)", 13.5, 3.0),
    ]
    combined_loss_block = ("Combined Loss\n(0.8 × CRF + 0.2 × CE)", training_x, 1.3)

    def draw_block(label, x, y, width=box_width, height=box_height, facecolor='white', fontsize=font_size):
        rect = mpatches.FancyBboxPatch(
            (x - width/2, y), width, height,
            boxstyle="round,pad=0.03",
            edgecolor='black', facecolor=facecolor, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y + height / 2, label, ha='center', va='center', fontsize=fontsize)

    # Draw top blocks
    for label, x, y in top_blocks:
        draw_block(label, x, y)

    # Draw pre-split blocks
    for label, x, y in pre_split_blocks:
        draw_block(label, x, y)

    # Draw parallel conv blocks
    for label, x, y in parallel_blocks:
        draw_block(label, x, y)

    # Draw concat block
    draw_block(concat_block[0], concat_block[1], concat_block[2])

    # Draw post-concat blocks
    for label, x, y in post_concat_blocks:
        draw_block(label, x, y)

    # Draw inference blocks (left side) with light blue background
    for label, x, y in inference_blocks:
        draw_block(label, x, y, facecolor='#e6f2ff')

    # Draw loss blocks (right side) with light red background
    for label, x, y in loss_blocks:
        draw_block(label, x, y, facecolor='#ffe6e6')

    # Draw combined loss block with light red background
    draw_block(combined_loss_block[0], combined_loss_block[1], combined_loss_block[2], facecolor='#ffe6e6')

    # === ARROWS ===

    # Arrows from inputs to first conv
    for top_block in top_blocks:
        ax.annotate('', xy=(pre_split_blocks[0][1], pre_split_blocks[0][2] + box_height),
                    xytext=(top_block[1], top_block[2]),
                    arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Arrows between pre-split blocks
    for i in range(len(pre_split_blocks) - 1):
        ax.annotate('', xy=(pre_split_blocks[i+1][1], pre_split_blocks[i+1][2] + box_height),
                    xytext=(pre_split_blocks[i][1], pre_split_blocks[i][2]),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Arrows from BiLSTM to parallel convs
    lstm_block = pre_split_blocks[-1]
    for parallel_block in parallel_blocks:
        ax.annotate('', xy=(parallel_block[1], parallel_block[2] + box_height),
                    xytext=(lstm_block[1], lstm_block[2]),
                    arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Arrows from parallel convs to concat
    for parallel_block in parallel_blocks:
        ax.annotate('', xy=(concat_block[1], concat_block[2] + box_height),
                    xytext=(parallel_block[1], parallel_block[2]),
                    arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Arrows between post-concat blocks
    ax.annotate('', xy=(post_concat_blocks[0][1], post_concat_blocks[0][2] + box_height),
                xytext=(concat_block[1], concat_block[2]),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    for i in range(len(post_concat_blocks) - 1):
        ax.annotate('', xy=(post_concat_blocks[i+1][1], post_concat_blocks[i+1][2] + box_height),
                    xytext=(post_concat_blocks[i][1], post_concat_blocks[i][2]),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))

    classifier_block = post_concat_blocks[-1]

    # Arrow from classifier to inference path (LEFT) - straight diagonal
    ax.annotate('', xy=(inference_blocks[0][1], inference_blocks[0][2] + box_height),
                xytext=(classifier_block[1] - box_width/4, classifier_block[2]),
                arrowprops=dict(facecolor='steelblue', arrowstyle='->', edgecolor='steelblue', linewidth=2))

    # Arrow between inference blocks
    ax.annotate('', xy=(inference_blocks[1][1], inference_blocks[1][2] + box_height),
                xytext=(inference_blocks[0][1], inference_blocks[0][2]),
                arrowprops=dict(facecolor='steelblue', arrowstyle='->', edgecolor='steelblue', linewidth=2))

    # Arrows from classifier to loss blocks (RIGHT) - straight diagonal
    ax.annotate('', xy=(loss_blocks[0][1], loss_blocks[0][2] + box_height),
                xytext=(classifier_block[1] + box_width/4, classifier_block[2]),
                arrowprops=dict(facecolor='firebrick', arrowstyle='->', edgecolor='firebrick', linewidth=2))

    ax.annotate('', xy=(loss_blocks[1][1], loss_blocks[1][2] + box_height),
                xytext=(classifier_block[1] + box_width/4, classifier_block[2]),
                arrowprops=dict(facecolor='firebrick', arrowstyle='->', edgecolor='firebrick', linewidth=2))

    # Arrows from loss blocks to combined loss
    for loss_block in loss_blocks:
        ax.annotate('', xy=(combined_loss_block[1], combined_loss_block[2] + box_height),
                    xytext=(loss_block[1], loss_block[2]),
                    arrowprops=dict(facecolor='firebrick', arrowstyle='->', edgecolor='firebrick',
                                    connectionstyle='arc3,rad=0', linewidth=2))

    # Add labels below the paths
    ax.text(inference_x, 0.3, "Inference", fontsize=16, fontweight='bold', color='steelblue', ha='center')
    ax.text(training_x, 0.3, "Training", fontsize=16, fontweight='bold', color='firebrick', ha='center')

    plt.title("SPCNNClassifier Architecture (T5 LSTM-CNN 6)", fontsize=20, fontweight='bold')
    plt.ylim(-0.3, 20.5)
    plt.xlim(-0.5, 16.5)
    plt.tight_layout()

    arch_path = Config.PLOTS_SAVE_DIR / 'model_architecture_6state_t5_lstm_cnn_6.png'
    plt.savefig(arch_path, dpi=150)
    print(f"Architecture diagram saved to: {arch_path}")

    return fig


def plot_all(results_path: Path = Config.TRAIN_VAL_LOSSES_PKL_SAVE_PATH):

    plot_training_metrics(results_path)
    plot_best_metrics_bar(results_path)
    draw_model_architecture()


if __name__ == "__main__":
    plot_all()



