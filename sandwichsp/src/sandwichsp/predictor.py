
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .config import Config
from .embeddings import T5Embedder
from .model import SPCNNClassifier
from .weights import ensure_weights


@dataclass
class PredictionResult:
    """For giving results of inference

    Attributes:
        sequence: The input amino acid sequence.
        labels: Per-residue label string (e.g., "SSSSSSIIIIII...").
        sp_type: High-level signal peptide type ("SP", "TAT", "LIPO", "NO_SP").
        cleavage_site: Position of cleavage site (0-indexed), or None if no SP.
    """

    sequence: str
    labels: str
    sp_type: str
    cleavage_site: Optional[int]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "labels": self.labels,
            "sp_type": self.sp_type,
            "cleavage_site": self.cleavage_site,
        }

    def __repr__(self) -> str:
        return (
            f"PredictionResult(sp_type={self.sp_type!r}, "
            f"cleavage_site={self.cleavage_site}, "
            f"seq_len={len(self.sequence)})"
        )


class SandwichSP:

    def __init__(
        self,
        device: str = "auto",
        weights_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto').
            weights_path: Path to model weights. If None, downloads automatically.
        """
        if device == "auto":
            device = Config.get_device()
        self.device = device

        # Initialize embedder (lazy loads T5 model)
        self._embedder = T5Embedder(device=device)

        # Load classifier weights
        if weights_path is None:
            weights_path = ensure_weights()
        else:
            weights_path = Path(weights_path)

        self._model = self._load_model(weights_path)

    def _load_model(self, weights_path: Path) -> SPCNNClassifier:
        """Load the classifier model with weights."""
        print(f"Loading SandwichSP classifier from {weights_path}...")

        model = SPCNNClassifier(
            embedding_dim=Config.EMBEDDING_DIM,
            num_labels=Config.NUM_LABELS,
            dropout=0.0,  # No dropout during inference
            lstm_hidden=Config.LSTM_HIDDEN,
            lstm_layers=Config.LSTM_LAYERS,
            conv_filters=Config.CONV_FILTERS,
        )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def predict(self, sequence: str) -> PredictionResult:
        """Predict signal peptide for a single sequence.

        Args:
            sequence: Amino acid sequence (e.g., "MKFLILLFNILCLFPVLAADNH...").

        Returns:
            PredictionResult with labels, SP type, and cleavage site.
        """
        # Generate embeddings
        embeddings = self._embedder.embed(sequence)
        embeddings = embeddings.unsqueeze(0)  # Add batch dimension
        attention_mask = torch.ones(1, embeddings.shape[1], dtype=torch.long, device=self.device)

        # Run prediction
        with torch.no_grad():
            predictions = self._model(embeddings, attention_mask)

        # Convert to labels
        pred_indices = predictions[0]  # Remove batch dimension
        labels = "".join(Config.LABEL_MAP_INV[idx] for idx in pred_indices)

        # Determine SP type and cleavage site
        sp_type, cleavage_site = self._analyze_prediction(labels)

        return PredictionResult(
            sequence=sequence,
            labels=labels,
            sp_type=sp_type,
            cleavage_site=cleavage_site,
        )

    def predict_batch(self, sequences: list[str]) -> list[PredictionResult]:
        """Predict signal peptides for multiple sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            List of PredictionResult objects.
        """
        if not sequences:
            return []

        # Generate embeddings
        embeddings, attention_mask = self._embedder.embed_batch(sequences)

        # Run prediction
        with torch.no_grad():
            predictions = self._model(embeddings, attention_mask)

        # Convert to results
        results = []
        for seq, pred_indices, mask in zip(sequences, predictions, attention_mask):
            # Only take labels for actual sequence (not padding)
            seq_len = mask.sum().item()
            labels = "".join(Config.LABEL_MAP_INV[idx] for idx in pred_indices[:seq_len])

            sp_type, cleavage_site = self._analyze_prediction(labels)

            results.append(
                PredictionResult(
                    sequence=seq,
                    labels=labels,
                    sp_type=sp_type,
                    cleavage_site=cleavage_site,
                )
            )

        return results

    def _analyze_prediction(self, labels: str) -> tuple[str, Optional[int]]:
        """Analyze prediction to determine SP type and cleavage site.

        Args:
            labels: Per-residue label string.

        Returns:
            Tuple of (sp_type, cleavage_site).
        """
        # Count label types in first 70 residues (typical SP region)
        sp_region = labels[:70]

        # Check for signal peptide labels
        sp_labels = {"S", "T", "L"}
        sp_count = sum(1 for c in sp_region if c in sp_labels)

        if sp_count == 0:
            return "NO_SP", None

        # Determine dominant SP type
        s_count = sp_region.count("S")
        t_count = sp_region.count("T")
        l_count = sp_region.count("L")

        if s_count >= t_count and s_count >= l_count:
            sp_type = "SP"
        elif t_count >= s_count and t_count >= l_count:
            sp_type = "TAT"
        else:
            sp_type = "LIPO"

        # Find cleavage site (transition from SP to non-SP)
        cleavage_site = None
        for i, label in enumerate(labels):
            if label in sp_labels:
                continue
            # Found first non-SP label
            if i > 0:
                cleavage_site = i  # 0-indexed position after last SP residue
            break

        return sp_type, cleavage_site

    @property
    def embedder(self) -> T5Embedder:
        """Access the underlying T5 embedder."""
        return self._embedder

    @property
    def model(self) -> SPCNNClassifier:
        """Access the underlying classifier model."""
        return self._model
