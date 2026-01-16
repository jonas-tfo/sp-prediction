"""T5 embedding generation for protein sequences."""

import re

import torch
from transformers import T5EncoderModel, T5Tokenizer

from .config import Config


class T5Embedder:
    """load T5 and generate protein embeddings
    """

    def __init__(self, device: str = None, half_precision: bool = True):
        """Initialize the T5 embedder.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto').
            half_precision: Use half precision (float16) for reduced memory.
        """
        if device is None or device == "auto":
            device = Config.get_device()
        self.device = device
        self.half_precision = half_precision and device != "cpu"

        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self) -> T5Tokenizer:
        """Lazily load the tokenizer."""
        if self._tokenizer is None:
            print(f"Loading tokenizer from {Config.T5_MODEL_NAME}...")
            self._tokenizer = T5Tokenizer.from_pretrained(
                Config.T5_MODEL_NAME,
                do_lower_case=False,
            )
        return self._tokenizer

    @property
    def model(self) -> T5EncoderModel:
        """Lazily load the T5 encoder model."""
        if self._model is None:
            print(f"Loading T5 model from {Config.T5_MODEL_NAME}...")
            print(f"Using device: {self.device}")
            self._model = T5EncoderModel.from_pretrained(Config.T5_MODEL_NAME)

            if self.half_precision:
                self._model = self._model.half()

            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess a protein sequence for T5.

        Args:
            sequence: Amino acid sequence (e.g., "MKFLIL...")

        Returns:
            Space-separated sequence with rare AAs replaced.
        """
        # Replace rare amino acids with X
        sequence = re.sub(r"[UZOB]", "X", sequence.upper())
        # Add spaces between amino acids (need to do this for the tokenizer)
        return " ".join(list(sequence))

    def embed(self, sequence: str) -> torch.Tensor:
        """Generate embeddings for a single sequence.

        Args:
            sequence: Amino acid sequence.

        Returns:
            Embedding tensor of shape (seq_len, 1024).
        """
        processed = self.preprocess_sequence(sequence)

        inputs = self.tokenizer(
            processed,
            return_tensors="pt",
            padding=False,
            add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        # Remove batch dimension and special tokens like the end of sequence one
        embeddings = embeddings[0, :-1, :]

        # Convert back to float32 if using half precision for classifier
        if self.half_precision:
            embeddings = embeddings.float()

        return embeddings

    def embed_batch(
        self,
        sequences: list[str],
        max_length: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate embeddings for a batch of sequences.

        Args:
            sequences: List of amino acid sequences.
            max_length: Maximum sequence length (None for auto).

        Returns:
            Tuple of (embeddings, attention_mask) tensors.
            embeddings: shape (batch, max_seq_len, 1024)
            attention_mask: shape (batch, max_seq_len)
        """
        processed = [self.preprocess_sequence(seq) for seq in sequences]

        inputs = self.tokenizer(
            processed,
            return_tensors="pt",
            padding=True,
            truncation=True if max_length else False,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        # Remove EOS token from each sequence
        embeddings = embeddings[:, :-1, :]
        attention_mask = attention_mask[:, :-1]

        # Convert back to float32 for classifier
        if self.half_precision:
            embeddings = embeddings.float()

        return embeddings, attention_mask
