
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class SPCNNClassifier(nn.Module):

    def __init__(self, embedding_dim: int = 1024, num_labels: int = 6):

        super().__init__()
        self.dropout = nn.Dropout(0.35)

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=1024,
            kernel_size=5,
            padding=2,
        )
        self.bn_conv = nn.BatchNorm1d(1024)

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(512 * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):

        hidden_states = embeddings.float() # has shape (batch_size, seq_len, embedding_dim)

        # Apply conv, batch normalization and ReLU
        x_conv = self.conv(hidden_states.transpose(1, 2)) # 1d cnn (batch_size, embedding_dim, seq_len)
        x_conv = self.bn_conv(x_conv)
        x_conv = F.relu_(x_conv)

        # Transpose for LSTM
        x_lstm_input = x_conv.transpose(1, 2)

        # Apply LSTM
        lstm_out, _ = self.lstm(x_lstm_input)

        # Classifier
        x_linear = self.classifier(lstm_out)
        logits = self.dropout(x_linear)

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


