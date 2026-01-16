import sys
from pathlib import Path
import torch
import torch.nn as nn

# --- Path Correction for Direct Execution ---
# Add the '6state' directory to the system path to allow absolute imports
# of modules within 't5_lstm_cnn_6', even when running this script directly.
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))
# --- End Path Correction ---

from t5_lstm_cnn_6.config import Config
from t5_lstm_cnn_6.model import SPCNNClassifier

def count_parameters(model: nn.Module):
    """Counts the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Instantiate the model using parameters from Config
    model = SPCNNClassifier(
        embedding_dim=Config.EMBEDDING_DIM,
        num_labels=Config.NUM_LABELS,
        dropout=Config.DROPOUT,
        lstm_hidden=Config.LSTM_HIDDEN,
        lstm_layers=Config.LSTM_LAYERS,
        conv_filters=Config.CONV_FILTERS,
    )

    num_params = count_parameters(model)
    # Format number with commas for readability
    print(f"Total number of trainable parameters in SPCNNClassifier: {num_params:,}")
