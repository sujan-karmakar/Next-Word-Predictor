import torch
from torch.utils.data import TensorDataset, DataLoader
from tokenization import tokenize_file, create_vocabulary, vectorize_lines, save_mappings

SEQ_LEN = 20
BATCH_SIZE = 64

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


def create_train_loader():
    tokenized_lines = tokenize_file("data/processed/combined_text.txt")
    word_to_token, token_to_word = create_vocabulary(tokenized_lines)
    vectorized_lines = vectorize_lines(tokenized_lines, word_to_token)

    save_mappings(word_to_token, token_to_word)

    X_train, Y_train = create_training_data(vectorized_lines, SEQ_LEN)

    print("X shape:", X_train.shape)
    print("Y shape:", Y_train.shape)


    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Number of batches:", len(train_loader))

    return train_loader, word_to_token, token_to_word