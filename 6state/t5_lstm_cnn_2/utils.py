"""Utility functions for data preparation and loading."""

import pandas as pd
from torch.utils.data import DataLoader

from .config import Config
from .dataset import SPDatasetWithEmbeddings


def ensure_dirs():
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_fold_paths(fold_num: int) -> dict:
    return {
        "train_csv": Config.DATA_PATH_FOLDS / f"fold_{fold_num}_train.csv",
        "val_csv": Config.DATA_PATH_FOLDS / f"fold_{fold_num}_val.csv",
        "train_emb": Config.DATA_PATH_FOLDS / f"fold_{fold_num}_train_embeddings.npz",
        "val_emb": Config.DATA_PATH_FOLDS / f"fold_{fold_num}_val_embeddings.npz",
    }


def prepare_fold_data(fold_num: int, batch_size: int = Config.BATCH_SIZE) -> tuple[DataLoader, DataLoader]:

    paths = Config.get_fold_paths(fold_num)

    train_dataset = SPDatasetWithEmbeddings(
        paths["train_csv"],
        paths["train_emb"],
    )
    val_dataset = SPDatasetWithEmbeddings(
        paths["val_csv"],
        paths["val_emb"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_validation_labels(fold_num: int) -> list:

    paths = Config.get_fold_paths(fold_num)
    val_df = pd.read_csv(paths["val_csv"])

    # Filter sequences with 'P' labels
    val_df_filtered = val_df[~val_df["labels"].str.contains("P", na=False)]
    val_df_encoded = val_df_filtered.copy()
    val_df_encoded["label"] = val_df_encoded["labels"].apply(
        lambda x: [Config.LABEL_MAP[c] for c in x if c in Config.LABEL_MAP]
    )
    val_df_encoded = val_df_encoded[val_df_encoded["label"].map(len) > 0]

    return val_df_encoded["label"].tolist()


def get_test_data() -> tuple[list, list, pd.DataFrame]:

    test_df = pd.read_csv(Config.TEST_CSV)

    # Filter sequences with 'P' labels
    test_df_filtered = test_df[~test_df["labels"].str.contains("P", na=False)]
    test_df_encoded = test_df_filtered.copy()
    test_df_encoded["label"] = test_df_encoded["labels"].apply(
        lambda x: [Config.LABEL_MAP[c] for c in x if c in Config.LABEL_MAP]
    )
    test_df_encoded = test_df_encoded[test_df_encoded["label"].map(len) > 0]

    test_seqs = test_df_encoded["sequence"].tolist()
    test_label_seqs = test_df_encoded["label"].tolist()

    return test_seqs, test_label_seqs, test_df_encoded
