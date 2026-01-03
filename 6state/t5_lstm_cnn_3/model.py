
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class SPCNNClassifier(nn.Module):

    def __init__(self, embedding_dim: int = 1024, num_labels: int = 6):

        super().__init__()
        self.dropout = nn.Dropout(0.35)

        self.conv9 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=256,
            kernel_size=9,
            padding=4,
        )
        self.conv7 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=256,
            kernel_size=9,
            padding=4,
        )
        self.conv5 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=256,
            kernel_size=5,
            padding=2,
        )
        self.bn_conv = nn.BatchNorm1d(1024)

        self.lstm = nn.LSTM(
            input_size=256 * 3,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True, # refers to the shape of input and output tensors
            dropout=0.2,
        )

        self.classifier = nn.Linear(512 * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):

        hidden_states = embeddings.float() # has shape (batch_size, seq_len, embedding_dim)

        # Apply 3 conv layers, batch normalization and ReLU
        transposed_hidden_states = hidden_states.transpose(1, 2) # transpose because 1d cnn needs (batch_size, embedding_dim, seq_len)

        conv5 = self.conv5(transposed_hidden_states)
        conv7 = self.conv7(transposed_hidden_states)
        conv9 = self.conv9(transposed_hidden_states)

        x_conv = torch.cat([conv5, conv7, conv9], dim=1) # concatenate along channel dimension

        # apply batch norm and GeLU
        x_conv = self.bn_conv(x_conv)
        x_conv = F.gelu(x_conv)

        # Transpose for LSTM
        x_lstm_input = x_conv.transpose(1, 2)

        # Apply LSTM
        lstm_out, _ = self.lstm(x_lstm_input)

        # apply dropout
        lstm_out = self.dropout(lstm_out)

        # Classifier
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


