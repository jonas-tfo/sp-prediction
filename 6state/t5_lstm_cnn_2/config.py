
import torch
from pathlib import Path


class Config:
    # Model
    MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
    EMBEDDING_DIM = 1024
    NUM_CLASSES = 6

    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.001
    NUM_FOLDS = 3
    PATIENCE = 4

    # Device
    DEVICE = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Main Working Dir Paths
    CWD = Path(__file__).resolve().parent

    # Output directory (within t5_lstm_cnn_2)
    OUTPUT_DIR = CWD / "outputs"

    # Data paths (for reading input data)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "aufgabe3"
    DATA_PATH_FOLDS = DATA_PATH / "3-fold"
    TEST_CSV = DATA_PATH / "reduced_30_signalP6_test.csv"
    TEST_EMBEDINGS = DATA_PATH / "reduced_30_signalP6_test_embeddings.npz"


    # Output paths (for saving results)
    PLOTS_SAVE_DIR = OUTPUT_DIR / "plots"
    TRAIN_VAL_LOSSES_PKL_SAVE_PATH = PLOTS_SAVE_DIR / "6state_t5_lstm_cnn_2_fold_results.pkl"

    # Model Saving Paths
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_SAVE_PATH = MODEL_DIR / "6state_t5_lstm_cnn_2.pt"
    MODEL_SAVE_PATH_TEMP = str(MODEL_DIR / "6state_t5_lstm_cnn_2_fold{}.pt")

    # Label mapping
    LABEL_MAP = {'S': 0, 'T': 1, 'L': 2, 'I': 3, 'M': 4, 'O': 5}
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
