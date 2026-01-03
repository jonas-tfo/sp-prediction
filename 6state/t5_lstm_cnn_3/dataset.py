
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config


class SPDatasetWithEmbeddings(Dataset):

    def __init__(self, csv_path: str | Path, embeddings_path: str):

        self.label_map = Config.LABEL_MAP
        self.df = pd.read_csv(csv_path)

        self.df = self.df[~self.df["labels"].str.contains("P", na=False)] # dont need pilin
        self.df["label"] = self.df["labels"].apply(
            lambda x: [self.label_map[c] for c in x if c in self.label_map]
        ) # numeric labels for classes
        self.df = self.df[self.df["label"].map(len) > 0] # filter out 0s

        # load embeddings keyed by uniprot_id
        self.embeddings = np.load(embeddings_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        uniprot_id = row["uniprot_id"]
        labels = row["label"]

        # Get embedding (seq_len, 1024) TODO make sure its being squeezed
        embedding = torch.tensor(self.embeddings[uniprot_id], dtype=torch.float32) # (seq_len, hidden_size)
        seq_len = embedding.shape[0]

        # Create labels tensor
        token_labels = list(labels) + [-100]  # -100 for EOS token
        while len(token_labels) < seq_len + 1:
            token_labels.append(-100)
        token_labels = token_labels[:seq_len]

        # Attention mask (all 1s for actual sequence)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        return {
            "embeddings": embedding,  # (seq_len, 1024)
            "attention_mask": attention_mask,  # (seq_len,) -> all ones for seq to train on
            "labels": torch.tensor(token_labels),
        }
