
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

from .config import Config

class SPCNNClassifier(nn.Module):

    def __init__(
          self,
          embedding_dim: int = Config.EMBEDDING_DIM,
          num_labels: int = Config.NUM_LABELS,
          dropout: float = Config.DROPOUT,
          lstm_hidden: int = Config.LSTM_HIDDEN,
          lstm_layers: int = Config.LSTM_LAYERS,
          conv_filters: int = Config.CONV_FILTERS,
        ):

        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.conv9 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_filters,
            kernel_size=9,
            padding=4,
        )

        self.bn_conv1 = nn.BatchNorm1d(conv_filters)

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True, # refers to the shape of input and output tensors
        )

        self.conv7 = nn.Conv1d(
            in_channels=embedding_dim,
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

        # Batch norm after concatenation of two conv layers
        self.bn_conv2 = nn.BatchNorm1d(conv_filters * 2)

        self.classifier = nn.Linear(conv_filters * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):

        hidden_states = embeddings.float() # has shape (batch_size, seq_len, embedding_dim)

        transposed_hidden_states = hidden_states.transpose(1, 2) # transpose because 1d cnn needs (batch_size, embedding_dim, seq_len)

        # Apply conv layer
        x_conv = self.conv5(transposed_hidden_states)

        # apply batch norm and GeLU
        x_conv = self.bn_conv1(x_conv)
        x_conv = F.gelu(x_conv)

        # Transpose back for LSTM
        x_lstm_input = x_conv.transpose(1, 2)

        # Apply LSTM
        lstm_out, _ = self.lstm(x_lstm_input)

        # Transpose back for conv layers
        x_lstm_out = lstm_out.transpose(1, 2)

        # apply two more conv layers
        conv7 = self.conv7(x_lstm_out)
        conv9 = self.conv9(x_lstm_out)

        # concatenate along channel dimension, shape is (batch_size, 256*2, seq_len)
        x_conv = torch.cat([conv7, conv9], dim=1)

        # apply batch norm and GeLU
        x_conv = self.bn_conv2(x_conv)
        x_conv = F.gelu(x_conv)

        # apply dropout before classifier
        lstm_out = self.dropout(x_conv.transpose(1, 2))

        # linear classifier to output layer
        logits = self.classifier(lstm_out)

        # loss during training (mean negative log likelihood)
        if labels is not None:
            crf_mask = (attention_mask.bool()) & (labels != -100)

            # Replace -100 with 0 for CRF compatibility
            mod_labels = labels.clone()
            mod_labels[labels == -100] = 0

            loss = -self.crf(logits, mod_labels, mask=crf_mask, reduction="mean")
            return loss
        # predictions during inference (Viterbi)
        else:
            # Decode only valid positions
            crf_mask = attention_mask.bool()
            predictions = self.crf.decode(logits, mask=crf_mask)
            return predictions


