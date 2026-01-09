
import torch
from pathlib import Path


class Config:
    # Model
    MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
    MODEL_NAME_SHORT = MODEL_NAME.split("/")[1]
    EMBEDDING_DIM = 1024
    NUM_CLASSES = 6
    DATA_PATH_EXTENSION = "_oversampled"

    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.001
    NUM_FOLDS = 3
    PATIENCE = 7

    # Device
    DEVICE = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Main Working Dir Paths
    CWD = Path(__file__).resolve().parent

    # Output directory (within t5_lstm_cnn_3)
    OUTPUT_DIR = CWD / "outputs"

    # Data paths (for reading input data)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "aufgabe3"
    DATA_PATH_FOLDS = DATA_PATH / f"{NUM_FOLDS}-fold"
    DATA_PATH_FOLDS_EMBEDDINGS = DATA_PATH / f"{NUM_FOLDS}-fold" / MODEL_NAME_SHORT
    TEST_CSV = DATA_PATH / f"reduced_30_signalP6_test{DATA_PATH_EXTENSION}.csv"
    TEST_EMBEDINGS = DATA_PATH / "embeddings" / MODEL_NAME_SHORT / f"reduced_30_signalP6_test_embeddings_{MODEL_NAME_SHORT}{DATA_PATH_EXTENSION}.npz"


    # Output paths (for saving results)
    PLOTS_SAVE_DIR = OUTPUT_DIR / "plots"
    TRAIN_VAL_LOSSES_PKL_SAVE_PATH = PLOTS_SAVE_DIR / "6state_t5_lstm_cnn_5_fold_results.pkl"

    # Model Saving Paths
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_SAVE_PATH = MODEL_DIR / "6state_t5_lstm_cnn_5.pt"
    MODEL_SAVE_PATH_TEMP = str(MODEL_DIR / "6state_t5_lstm_cnn_5_fold{}.pt")

    # Label mapping
    LABEL_MAP = {'S': 0, 'T': 1, 'L': 2, 'I': 3, 'M': 4, 'O': 5}
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
