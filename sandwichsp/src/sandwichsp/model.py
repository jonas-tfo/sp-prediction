"""Neural network model for signal peptide prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .config import Config


# this is jsut for the cross entropy class weighting
def inverse_freq_weights(dist: dict, label_map: dict) -> torch.Tensor:
    """Compute class weights using inverse frequency weighting."""
    weights = torch.zeros(len(label_map))
    for label, idx in label_map.items():
        weights[idx] = 1.0 / dist[label]
    weights = weights / weights.sum() * len(label_map)
    return weights


class SPCNNClassifier(nn.Module):

    def __init__(
        self,
        embedding_dim: int = Config.EMBEDDING_DIM,
        num_labels: int = Config.NUM_LABELS,
        dropout: float = Config.DROPOUT,
        lstm_hidden: int = Config.LSTM_HIDDEN,
        lstm_layers: int = Config.LSTM_LAYERS,
        conv_filters: int = Config.CONV_FILTERS,
        class_weights: torch.Tensor = None,
        ce_weight: float = 0.2,
    ):
        super().__init__()

        if class_weights is None:
            class_weights = inverse_freq_weights(Config.LABEL_DIST, Config.LABEL_MAP)

        self.ce_weight = ce_weight
        self.dropout = nn.Dropout(dropout)

        self.conv9 = nn.Conv1d(
            in_channels=lstm_hidden * 2,
            out_channels=conv_filters,
            kernel_size=9,
            padding=4,
        )
        self.bn_conv1 = nn.BatchNorm1d(conv_filters)

        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.conv7 = nn.Conv1d(
            in_channels=lstm_hidden * 2,
            out_channels=conv_filters,
            kernel_size=7,
            padding=3,
        )

        self.conv5 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_filters,
            kernel_size=5,
            padding=2,
        )

        self.bn_conv2 = nn.BatchNorm1d(conv_filters * 2)
        self.classifier = nn.Linear(conv_filters * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100,
            reduction="mean",
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            embeddings: Input embeddings of shape (batch, seq_len, embedding_dim)
            attention_mask: Mask of shape (batch, seq_len)
            labels: Optional labels for training, shape (batch, seq_len)

        Returns:
            If labels provided: loss value
            If no labels: list of predicted label sequences
        """
        hidden_states = embeddings.float()
        transposed_hidden_states = hidden_states.transpose(1, 2)

        x_conv = self.conv5(transposed_hidden_states)
        x_conv = self.bn_conv1(x_conv)
        x_conv = F.gelu(x_conv)

        x_lstm_input = x_conv.transpose(1, 2)
        lstm_out, _ = self.lstm(x_lstm_input)
        x_lstm_out = lstm_out.transpose(1, 2)

        conv7 = self.conv7(x_lstm_out)
        conv9 = self.conv9(x_lstm_out)

        x_conv = torch.cat([conv7, conv9], dim=1)
        x_conv = self.bn_conv2(x_conv)
        x_conv = F.gelu(x_conv)

        lstm_out = self.dropout(x_conv.transpose(1, 2))
        logits = self.classifier(lstm_out)

        if labels is not None:
            # Training mode: compute loss
            crf_mask = (attention_mask.bool()) & (labels != -100)
            mod_labels = labels.clone()
            mod_labels[labels == -100] = 0
            crf_loss = -self.crf(logits, mod_labels, mask=crf_mask, reduction="mean")

            ce_loss = self.ce_loss(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            total_loss = (1 - self.ce_weight) * crf_loss + self.ce_weight * ce_loss
            return total_loss
        else:
            # Inference mode: decode predictions
            crf_mask = attention_mask.bool()
            # decode logits with viterbi
            predictions = self.crf.decode(logits, mask=crf_mask)
            return predictions
