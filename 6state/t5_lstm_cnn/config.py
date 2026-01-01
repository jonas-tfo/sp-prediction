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

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "aufgabe3"
    DATA_PATH_FOLDS = DATA_PATH / "3-fold"

    MODEL_DIR = BASE_DIR / "models" / "6state_t5_lstm_cnn"
    MODEL_SAVE_PATH = MODEL_DIR / "6state_t5_lstm_cnn.pt"
    MODEL_SAVE_PATH_TEMP = str(BASE_DIR / "models" / "6state_t5_lstm_cnn_fold{}.pt")

    OUTPUT_DIR = DATA_PATH / "6state_t5_lstm_cnn" / "outputs"

    TEST_CSV = DATA_PATH / "reduced_30_signalP6_test.csv"

    # Label mapping
    LABEL_MAP = {'S': 0, 'T': 1, 'L': 2, 'I': 3, 'M': 4, 'O': 5}
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

    @classmethod
    def ensure_dirs(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_fold_paths(cls, fold_num: int) -> dict:
        return {
            "train_csv": cls.DATA_PATH_FOLDS / f"fold_{fold_num}_train.csv",
            "val_csv": cls.DATA_PATH_FOLDS / f"fold_{fold_num}_val.csv",
            "train_emb": cls.DATA_PATH_FOLDS / f"fold_{fold_num}_train_embeddings.npz",
            "val_emb": cls.DATA_PATH_FOLDS / f"fold_{fold_num}_val_embeddings.npz",
        }
