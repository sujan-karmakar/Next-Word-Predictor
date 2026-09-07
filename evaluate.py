import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tokenization import tokenize_file, vectorize_lines, load_mappings
from data_loader import create_training_data
from model import GRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
BATCH_SIZE = 64

MODEL_PATH = "best_model.pth"

def create_test_loader(file_path, word_to_token):
    tokenized_lines = tokenize_file(file_path)
    vectorized_lines = vectorize_lines(tokenized_lines, word_to_token)

    X_test, Y_test = create_training_data(vectorized_lines, SEQ_LEN)

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader

word_to_token, token_to_word = load_mappings("word_mapping")
vocab_size = len(word_to_token)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = GRU(
    vocab_size = vocab_size,
    embedding_dim = checkpoint["embedding_dim"],
    hidden_size = checkpoint["hidden_size"],
    num_layers = checkpoint["num_layers"],
    dropout = checkpoint["dropout"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

criterion = nn.CrossEntropyLoss()

def evaluate_model(test_loader):
    total_loss = 0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == Y_batch).sum().item()

    average_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(average_loss))
    accuracy = correct / total_samples

    return average_loss, perplexity.item(), accuracy

ptb_test_loader = create_test_loader("data/test/ptb_test.txt", word_to_token)
wikitext_test_loader = create_test_loader("data/test/wikitext2_test.txt", word_to_token)

ptb_loss, ptb_perplexity, ptb_accuracy = evaluate_model(ptb_test_loader)
wikitext_loss, wikitext_perplexity, wikitext_accuracy = evaluate_model(wikitext_test_loader)

print("Device:", DEVICE)
print("Vocabulary size:", vocab_size)
print()
print("PTB Test Results")
print("Loss:", f"{ptb_loss:.4f}")
print("Perplexity:", f"{ptb_perplexity:.4f}")
print("Accuracy:", f"{ptb_accuracy * 100:.2f}%")
print()
print("WikiText-2 Test Results")
print("Loss:", f"{wikitext_loss:.4f}")
print("Perplexity:", f"{wikitext_perplexity:.4f}")
print("Accuracy:", f"{wikitext_accuracy * 100:.2f}%")