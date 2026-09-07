import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=192, hidden_size=384, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx = 0
        )

        self.gru = nn.GRU(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.fc(self.dropout_layer(hidden[-1]))
