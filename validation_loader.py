from torch.utils.data import TensorDataset, DataLoader
from tokenization import tokenize_file, vectorize_lines, load_mappings
from data_loader import create_training_data

SEQ_LEN = 20
BATCH_SIZE = 64

def create_validation_loader():
    word_to_token, token_to_word = load_mappings("word_mapping")
    tokenized_lines = tokenize_file("data/validation/ptb_valid.txt")
    vectorized_lines = vectorize_lines(tokenized_lines, word_to_token)

    X_val, Y_val = create_training_data(vectorized_lines, SEQ_LEN)

    print("Validation X shape:", X_val.shape)
    print("Validation Y shape:", Y_val.shape)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Number of validation batches:", len(val_loader))
    
    return val_loader