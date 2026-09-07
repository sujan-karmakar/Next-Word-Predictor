import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2, dropout=0.2):
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

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.gru(embedded)

        last_output = output[:, -1, :]

        logits = self.fc(last_output)

        return logits