
import torch


class Config:
    # Model architecture
    EMBEDDING_DIM: int = 1024
    NUM_LABELS: int = 6
    LSTM_HIDDEN: int = 512
    LSTM_LAYERS: int = 2
    CONV_FILTERS: int = 256
    DROPOUT: float = 0.126

    # Label mapping
    LABEL_MAP = {"S": 0, "T": 1, "L": 2, "I": 3, "M": 4, "O": 5}
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

    # Label descriptions
    LABEL_DESCRIPTIONS = {
        "S": "Sec/SPI signal peptide",
        "T": "Tat/SPI signal peptide",
        "L": "Sec/SPII signal peptide (lipoprotein)",
        "I": "Cytoplasm (no signal peptide)",
        "M": "Transmembrane region",
        "O": "Other/extracellular",
    }

    # SP type mapping (for high-level classification)
    SP_TYPE_MAP = {
        "S": "SP",      # Sec signal peptide
        "T": "TAT",     # Tat signal peptide
        "L": "LIPO",    # Lipoprotein signal peptide
        "I": "NO_SP",   # No signal peptide
        "M": "NO_SP",   # No signal peptide
        "O": "OTHER",   # Other
    }

    # Class distribution (for weighted loss during training)
    LABEL_DIST = {
        "S": 0.05,
        "T": 0.02,
        "L": 0.03,
        "I": 0.65,
        "M": 0.05,
        "O": 0.20,
    }

    # T5 embedding model (half precision t5)
    T5_MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"

    # Weights download
    WEIGHTS_URL = "https://github.com/jonas-tfo/sp-prediction/releases/download/v.0.1.0/sandwichsp.pt"
    WEIGHTS_FILENAME = "sandwichsp_weights.pt"

    @staticmethod
    def get_device() -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
