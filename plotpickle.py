import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the saved losses data
losses_pickle_path = "/Users/jonas/Desktop/Uni/PBL/sp-prediction/data/aufgabe3/outputs/6state_ohe_lstm_cnn/6state_ohe_lstm_cnn_train_val_losses.pkl"

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

# Create figure with subplots in two rows
ncols = int(np.ceil(num_folds / 2))
fig, axes = plt.subplots(2, ncols, figsize=(15, 8), facecolor='white')
fig.suptitle('Training and Validation Losses (One Hot Encoding Variant)',
             fontsize=16, fontweight='bold', color='black')

# Flatten axes array for easier indexing
axes_flat = axes.flatten()

# Plot individual folds
for i in range(num_folds):
    ax = axes_flat[i]
    
    ax.set_facecolor('white')
    
    epochs = range(1, len(train_losses[i]) + 1)
    
    ax.plot(epochs, train_losses[i], color='darkorange', marker='o', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, val_losses[i], color='dodgerblue', marker='s', label='Val Loss', linewidth=2, markersize=4)
    
    # Mark best epoch with star only
    best_epoch = np.argmin(val_losses[i]) + 1
    best_val_loss = min(val_losses[i])
    ax.plot(best_epoch, best_val_loss, color='black', marker='*', markersize=15,
            label=f'Best: {best_val_loss:.4f}', zorder=5)
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=10, color='black')
    ax.set_ylabel('Loss', fontsize=10, color='black')
    ax.set_title(f'Fold {fold_numbers[i]}' + (' (Best Model)' if fold_numbers[i] == best_fold_num else ''), 
                 fontsize=12, color='black', fontweight='bold' if fold_numbers[i] == best_fold_num else 'normal')
    ax.legend(loc='upper right', fontsize=10, facecolor='white', edgecolor='gray')
    
    # Add horizontal grid lines only
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5, axis='y')
    
    # Make tick labels black
    ax.tick_params(colors='black')

# Hide any unused subplots
for i in range(num_folds, len(axes_flat)):
    axes_flat[i].axis('off')

plt.tight_layout()
plt.show()
