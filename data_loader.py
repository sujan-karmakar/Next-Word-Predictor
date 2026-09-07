import torch
from torch.utils.data import TensorDataset, DataLoader
from tokenization import BOS_TOKEN, EOS_TOKEN, tokenize_file, create_vocabulary, vectorize_lines, save_mappings

SEQ_LEN = 20
BATCH_SIZE = 64

def create_training_data(vectorized_lines, seq_len, bos_token, eos_token):
    X = []
    Y, lengths = [], []

    for line in vectorized_lines:
        tokens = [bos_token] + line + [eos_token]
        for i in range(1, len(tokens)):
            sequence = tokens[max(0, i - seq_len):i]
            target = tokens[i]
            lengths.append(len(sequence))
            sequence = sequence + [0] * (seq_len - len(sequence))

            X.append(sequence)
            Y.append(target)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)

    return X, Y, torch.tensor(lengths, dtype=torch.long)


def create_train_loader():
    tokenized_lines = tokenize_file("data/processed/combined_text.txt")
    word_to_token, token_to_word = create_vocabulary(tokenized_lines)
    vectorized_lines = vectorize_lines(tokenized_lines, word_to_token)

    save_mappings(word_to_token, token_to_word)

    X_train, Y_train, lengths = create_training_data(vectorized_lines, SEQ_LEN, word_to_token[BOS_TOKEN], word_to_token[EOS_TOKEN])

    print("X shape:", X_train.shape)
    print("Y shape:", Y_train.shape)


    train_dataset = TensorDataset(X_train, Y_train, lengths)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())

    print("Number of batches:", len(train_loader))

    return train_loader, word_to_token, token_to_word
