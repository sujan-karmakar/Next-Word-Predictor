import torch


def create_training_data(vectorized_lines, seq_len):
    X = []
    Y = []

    for line in vectorized_lines:
        for i in range(1, len(line)):
            sequence = line[max(0, i - seq_len):i]
            target = line[i]

            sequence = [0] * (seq_len - len(sequence)) + sequence

            X.append(sequence)
            Y.append(target)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)

    return X, Y