
from tqdm import utils
import torch
from pathlib import Path


class Config:
    # Model
    MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
    MODEL_NAME_SHORT = MODEL_NAME.split("/")[1]
    EMBEDDING_DIM = 1024
    NUM_CLASSES = 6
    DATA_PATH_EXTENSION = ""

    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.0006647135865318024
    NUM_FOLDS = 5
    PATIENCE = 7
    EMBEDDING_DIM: int = 1024
    NUM_LABELS: int = 6
    DROPOUT: float = 0.12602063719411183
    WEIGHT_DECAY: float = 0.00032476735706274504
    LSTM_HIDDEN: int = 512
    LSTM_LAYERS: int = 2
    CONV_FILTERS: int = 256

    # Label mapping
    LABEL_MAP = {'S': 0, 'T': 1, 'L': 2, 'I': 3, 'M': 4, 'O': 5}
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

    # Weighting for combined loss
    CE_WEIGHT: float = 0.2
    LABEL_DIST = {
        'S': 0.05,
        'T': 0.02,
        'L': 0.03,
        'I': 0.65,
        'M': 0.05,
        'O': 0.20,
    }
    CLASS_WEIGHTS: torch.Tensor = None

    # Device
    DEVICE = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Main Working Dir Paths
    CWD = Path(__file__).resolve().parent

    # Output directory (within t5_lstm_cnn_6)
    OUTPUT_DIR = CWD / "outputs"

    # Data paths (for reading input data)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "aufgabe3"
    DATA_PATH_FOLDS = DATA_PATH / f"{NUM_FOLDS}-fold"
    DATA_PATH_FOLDS_EMBEDDINGS = DATA_PATH / f"{NUM_FOLDS}-fold" / MODEL_NAME_SHORT
    TEST_CSV = DATA_PATH / f"reduced_30_signalP6_test_filtered_final{DATA_PATH_EXTENSION}.csv"
    TEST_EMBEDINGS = DATA_PATH / MODEL_NAME_SHORT / f"reduced_30_signalP6_test_embeddings_{MODEL_NAME_SHORT}{DATA_PATH_EXTENSION}.npz"
    TEST_EMBEDINGS_FILTERED = DATA_PATH / MODEL_NAME_SHORT / f"reduced_30_signalP6_test_embeddings_{MODEL_NAME_SHORT}{DATA_PATH_EXTENSION}_filtered.npz"


    # Output paths (for saving results)
    PLOTS_SAVE_DIR = OUTPUT_DIR / "plots"
    TRAIN_VAL_LOSSES_PKL_SAVE_PATH = PLOTS_SAVE_DIR / "6state_t5_lstm_cnn_6_fold_results.pkl"

    # Model Saving Paths
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_SAVE_PATH = MODEL_DIR / "6state_t5_lstm_cnn_6.pt"
    MODEL_SAVE_PATH_TEMP = str(MODEL_DIR / "6state_t5_lstm_cnn_5_fold{}.pt")
