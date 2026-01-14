import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_models_comparison(model_files, model_names=None, save_path='all_models_comparison.png'):
    """
    Plots a comparison of metrics from multiple model result pickle files.

    Args:
        model_files (list): List of paths to .pkl files containing fold results.
        model_names (list, optional): List of names for the models. If None, filenames are used.
        save_path (str): Path to save the resulting plot.
    """
    if model_names is None:
        model_names = [Path(f).stem for f in model_files]
    
    if len(model_files) != len(model_names):
        print(f"Error: Number of model files ({len(model_files)}) and names ({len(model_names)}) must match.")
        return

    # Data container
    plot_data = {
        'names': [],
        'MCC_mean': [], 'MCC_std': [],
        'Seq_Acc_mean': [], 'Seq_Acc_std': [],
        'Seq_Acc_SP_mean': [], 'Seq_Acc_SP_std': []
    }

    print("Loading data...")
    for i, file_path in enumerate(model_files):
        path_obj = Path(file_path)
        if not path_obj.exists():
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue

        try:
            with open(path_obj, 'rb') as f:
                data = pickle.load(f)
            
            if 'best_metrics' not in data:
                print(f"Warning: 'best_metrics' key missing in {file_path}. Skipping.")
                continue

            all_fold_metrics = data['best_metrics']
            
            # Extract values from each fold
            mccs = [m['mcc'] for m in all_fold_metrics]
            seq_accs = [m['seq_acc'] for m in all_fold_metrics]
            # Handle potential key difference or missing key safely if needed, 
            # but we confirmed seq_acc_only_sps exists in the requested files.
            seq_accs_sp = [m.get('seq_acc_only_sps', 0) for m in all_fold_metrics]

            # Store stats
            plot_data['names'].append(model_names[i])
            
            plot_data['MCC_mean'].append(np.mean(mccs))
            plot_data['MCC_std'].append(np.std(mccs))
            
            plot_data['Seq_Acc_mean'].append(np.mean(seq_accs))
            plot_data['Seq_Acc_std'].append(np.std(seq_accs))
            
            plot_data['Seq_Acc_SP_mean'].append(np.mean(seq_accs_sp))
            plot_data['Seq_Acc_SP_std'].append(np.std(seq_accs_sp))
            
            print(f"Loaded {model_names[i]}: MCC={np.mean(mccs):.3f}, SeqAcc={np.mean(seq_accs):.3f}, SeqAccSP={np.mean(seq_accs_sp):.3f}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    n_models = len(plot_data['names'])
    if n_models == 0:
        print("No valid data loaded. Exiting.")
        return

    print(f"Plotting comparison for {n_models} models...")

    # Plot settings
    metric_names = ['MCC', 'Sequence Accuracy', 'Sequence Accuracy (SP only)']
    n_metrics = len(metric_names)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate bar positions
    # Group by metric
    x = np.arange(n_metrics)
    total_width = 0.8
    bar_width = total_width / n_models
    
    # Use a colormap
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_models))

    for i in range(n_models):
        means = [
            plot_data['MCC_mean'][i],
            plot_data['Seq_Acc_mean'][i],
            plot_data['Seq_Acc_SP_mean'][i]
        ]
        stds = [
            plot_data['MCC_std'][i],
            plot_data['Seq_Acc_std'][i],
            plot_data['Seq_Acc_SP_std'][i]
        ]
        
        # Calculate offset for this model's bar
        offset = (i * bar_width) - (total_width / 2) + (bar_width / 2)
        
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, label=plot_data['names'][i],
               capsize=4, color=colors[i], edgecolor='white', alpha=0.9, zorder=3)
        
        # Add error annotations
        for j, rect in enumerate(bars):
            height = rect.get_height()
            std_val = stds[j]
            ax.text(rect.get_x() + rect.get_width()/2., height + std_val + 0.01,
                    f'±{std_val:.3f}',
                    ha='center', va='bottom', fontsize=9, rotation=0, zorder=5)

    # Styling
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison (Mean ± Standard Deviation)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=14)
    ax.legend(title="Models", loc='lower right', bbox_to_anchor=(1.25, 0.5), fontsize=10, title_fontsize=12, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {Path(save_path).resolve()}")

if __name__ == "__main__":
    base_path = Path("/home/jonas/Documents/Projects/sp-prediction")
    
    # Define the list of result pickle files
    model_files = [
        base_path / "data/aufgabe3/outputs/train_val_losses_ohe.pkl",
        base_path / "data/aufgabe3/outputs/train_val_losses_vhse.pkl",
        base_path / "6state/t5_lstm_cnn_6/outputs/plots/6state_t5_lstm_cnn_6_fold_results.pkl",
    ]
    
    # Define readable names for the models
    model_names = [
        "One-Hot Encoding",
        "VHSE Encoding",
        "Final Model \n (ProtT5 Embeddings)",
    ]
    
    output_file = "all_models_comparison.png"
    
    plot_models_comparison(model_files, model_names, save_path=output_file)
