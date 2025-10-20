import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the saved losses data
losses_pickle_path = "/Users/jonas/Downloads/train_val_losses.pkl"

with open(losses_pickle_path, 'rb') as f:
    losses_data = pickle.load(f)
    print(f"Loaded losses data from {losses_pickle_path}")

# Extract data
fold_numbers = losses_data['fold_numbers']
train_losses = losses_data['train_losses']
val_losses = losses_data['val_losses']
best_val_losses = losses_data['best_val_losses']
best_fold_num = losses_data['best_fold_num']
num_folds = losses_data['num_folds']

print(f"Loaded losses data:")
print(f"  Number of folds: {num_folds}")
print(f"  Best fold: {best_fold_num}")
print(f"  Epochs per fold: {[len(train_losses[i]) for i in range(num_folds)]}")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#121212')
fig.suptitle('Training and Validation Losses - K-Fold Cross Validation',
             fontsize=16, fontweight='bold', color='white')

# Plot individual folds
for i in range(num_folds):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    ax.set_facecolor('#1e1e1e')
    
    epochs = range(1, len(train_losses[i]) + 1)
    
    ax.plot(epochs, train_losses[i], color='orange', marker='o', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, val_losses[i], color='dodgerblue', marker='s', label='Val Loss', linewidth=2, markersize=4)
    
    # Mark best epoch
    best_epoch = np.argmin(val_losses[i]) + 1
    best_val_loss = min(val_losses[i])
    ax.axvline(x=best_epoch, color='white', linestyle='--', alpha=0.4, label=f'Best Epoch {best_epoch}')
    ax.plot(best_epoch, best_val_loss, color='white', marker='*', markersize=10,
            label=f'Best Val: {best_val_loss:.4f}')
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=10, color='white')
    ax.set_ylabel('Loss', fontsize=10, color='white')
    ax.set_title(f'Fold {fold_numbers[i]}' + (' (Best Model)' if fold_numbers[i] == best_fold_num else ''), 
                 fontsize=12, color='white', fontweight='bold' if fold_numbers[i] == best_fold_num else 'normal')
    ax.legend(loc='upper right', fontsize=8, facecolor='#2c2c2c', edgecolor='none', labelcolor='white')
    
    # Remove default grid, add white horizontal lines instead
    ax.grid(False)
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        ax.axhline(y=y, color='white', alpha=0.1, linewidth=0.8)
    
    # Make tick labels white
    ax.tick_params(colors='white')

# Add overall statistics subplot (if you have 5 folds, use the 6th subplot space)
if num_folds == 5:
    ax_stats = axes[1, 2]
    ax_stats.set_facecolor('#1e1e1e')
    ax_stats.axis('off')
    
    stats_text = f"""
    Cross-Validation Summary
    
    Number of Folds: {num_folds}
    
    Best Fold: {best_fold_num}
    Best Val Loss: {losses_data['best_fold_loss']:.4f}
    
    Average Best Val Loss: {losses_data['avg_best_val_loss']:.4f}
    
    Best Val Loss per Fold:
    """
    
    for i, (fold, loss) in enumerate(zip(fold_numbers, best_val_losses), 1):
        stats_text += f"\n    Fold {fold}: {loss:.4f}"
        if fold == best_fold_num:
            stats_text += " ★"
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                  family='monospace', color='white',
                  bbox=dict(boxstyle='round', facecolor='#2c2c2c', alpha=0.4))

plt.tight_layout()
plt.show()
