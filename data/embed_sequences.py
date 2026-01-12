import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from pathlib import Path


# setup variables
# "Rostlab/prot_t5_xl_half_uniref50-enc" (~1.2B params)
# "Rostlab/prot_t5_base_mt_uniref50" (~220M params)
# MODEL_NAME: str = "Rostlab/prot_t5_xl_half_uniref50-enc"
MODEL_NAME: str = "Rostlab/prot_t5_xl_half_uniref50-enc"
MODEL_NAME_SHORT: str = MODEL_NAME.split("/")[1]
NUM_FOLDS: int = 1
DATA_PATH_EXTENSION: str = ""

DEVICE: str = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "data" / "aufgabe3"
DATA_PATH_FOLDS: Path = DATA_PATH / f"{NUM_FOLDS}-fold"
TEST_CSV = DATA_PATH / f"reduced_30_signalP6_test{DATA_PATH_EXTENSION}.csv"
TEST_EMBEDINGS = DATA_PATH / MODEL_NAME_SHORT / f"reduced_30_signalP6_test_embeddings_{MODEL_NAME_SHORT}{DATA_PATH_EXTENSION}.npz"

print(f"Project base directory set to: {BASE_DIR}")
print(f"Data directory set to: {DATA_PATH}")
print(f"Folds data path set to: {DATA_PATH_FOLDS}")
print(f"Data path set to: {DATA_PATH}")
print(f"Using model: {MODEL_NAME}")


def embed_sequence(sequence, tokenizer: T5Tokenizer, encoder: T5EncoderModel, device: str = DEVICE) -> torch.Tensor:

    # spaces between needed for tokenizer
    seq_spaced = " ".join(list(sequence))

    # tokenize
    tokenized = tokenizer(
        seq_spaced, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # Get embeddings
    with torch.no_grad():
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state  # (1, seq_len, hidden_dim)

    # Remove batch dimension
    embeddings = embeddings.squeeze(0)  # (seq_len, hidden_dim)

    # need to exclude the end of sequence token
    seq_len = len(sequence)
    embeddings = embeddings[:seq_len]  # (seq_len, hidden_dim)

    return embeddings.float()  # (seq_len, hidden_dim)


def embed_train_val_data():
    # load the transformer and the tokenizer
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    encoder: T5EncoderModel = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    encoder.to(DEVICE)

    # Generate and save embeddings for each fold's train/val sets
    for fold_idx in range(NUM_FOLDS):
        # load data for according fold
        fold_train = pd.read_csv(DATA_PATH_FOLDS / f"fold_{fold_idx + 1}_train{DATA_PATH_EXTENSION}.csv")
        fold_val = pd.read_csv(DATA_PATH_FOLDS / f"fold_{fold_idx + 1}_val{DATA_PATH_EXTENSION}.csv")

        # go through train and val data
        for df, split in [(fold_train, "train"), (fold_val, "val")]:
            embeddings_dict = {}
            print(f"Embedding sequences for fold {fold_idx + 1} {split} set:")

            for idx, row in df.iterrows():
                uniprot_id = row["uniprot_id"]
                sequence = row["sequence"]
                embedding = embed_sequence(sequence, tokenizer, encoder)  # embed per residue
                embeddings_dict[uniprot_id] = embedding.cpu().numpy()  # save to dict as numpy array for later use for npz saving

                if (len(embeddings_dict)) % 100 == 0:
                    print(f"  Processed {len(embeddings_dict)}/{len(df)} sequences")

            # save embeddings to npz file
            save_path = DATA_PATH_FOLDS / MODEL_NAME_SHORT / f"fold_{fold_idx + 1}_{split}_embeddings{DATA_PATH_EXTENSION}.npz"
            np.savez(save_path, **embeddings_dict) # each key value pair passed seperately, key as array identifier, value as contents in npz
            print(f"  Saved {len(embeddings_dict)} embeddings to {save_path.name}")


def embed_test_data():
    # load the transformer and the tokenizer
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    encoder: T5EncoderModel = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    encoder.to(DEVICE)

    test_df = pd.read_csv(TEST_CSV)

    embeddings_dict = {}
    print(f"Embedding sequences for test set:")

    for idx, row in test_df.iterrows():
        uniprot_id = row["uniprot_id"]
        sequence = row["sequence"]
        embedding = embed_sequence(sequence, tokenizer, encoder)  # embed per residue
        embeddings_dict[uniprot_id] = embedding.cpu().numpy()  # save to dict as numpy array for later use for npz saving

        if (len(embeddings_dict)) % 100 == 0:
            print(f"  Processed {len(embeddings_dict)}/{len(test_df)} sequences")

    # save embeddings to npz file
    save_path = TEST_EMBEDINGS
    np.savez(save_path, **embeddings_dict) # each key value pair passed seperately, key as array identifier, value as contents in npz
    print(f"  Saved {len(embeddings_dict)} embeddings to {save_path.name}")

if __name__ == "__main__":
    embed_train_val_data()



