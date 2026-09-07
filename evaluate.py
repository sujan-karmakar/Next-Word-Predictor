import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tokenization import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, tokenize_file, vectorize_lines, load_mappings, vocabulary_fingerprint
from data_loader import create_training_data
from model import GRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
BATCH_SIZE = 64

MODEL_PATH = "best_model.pth"

def create_test_loader(file_path, word_to_token):
    tokenized_lines = tokenize_file(file_path)
    vectorized_lines = vectorize_lines(tokenized_lines, word_to_token)

    X_test, Y_test, lengths = create_training_data(vectorized_lines, SEQ_LEN, word_to_token[BOS_TOKEN], word_to_token[EOS_TOKEN])

    test_dataset = TensorDataset(X_test, Y_test, lengths)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader

word_to_token, token_to_word = load_mappings("word_mapping")
vocab_size = len(word_to_token)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if checkpoint.get("vocabulary_fingerprint") != vocabulary_fingerprint(word_to_token):
    raise RuntimeError("Checkpoint and word_mapping do not match. Run preprocessing.py and train.py together before evaluation.")

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
    correct = known_correct = known_samples = unknown_targets = 0

    with torch.no_grad():
        for X_batch, Y_batch, lengths in test_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model(X_batch, lengths)
            loss = criterion(logits, Y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == Y_batch).sum().item()
            known = Y_batch != word_to_token[UNK_TOKEN]
            known_samples += known.sum().item()
            known_correct += ((predictions == Y_batch) & known).sum().item()
            unknown_targets += (~known).sum().item()

    average_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(average_loss))
    accuracy = correct / total_samples

    return average_loss, perplexity.item(), accuracy, known_correct / max(known_samples, 1), unknown_targets / total_samples

ptb_test_loader = create_test_loader("data/test/ptb_test.txt", word_to_token)
wikitext_test_loader = create_test_loader("data/test/wikitext2_test.txt", word_to_token)

ptb_loss, ptb_perplexity, ptb_accuracy, ptb_known_accuracy, ptb_unk_rate = evaluate_model(ptb_test_loader)
wikitext_loss, wikitext_perplexity, wikitext_accuracy, wiki_known_accuracy, wiki_unk_rate = evaluate_model(wikitext_test_loader)

print("Device:", DEVICE)
print("Vocabulary size:", vocab_size)
print()
print("PTB Test Results")
print("Loss:", f"{ptb_loss:.4f}")
print("Perplexity:", f"{ptb_perplexity:.4f}")
print("Accuracy:", f"{ptb_accuracy * 100:.2f}%")
print("Known-token accuracy:", f"{ptb_known_accuracy * 100:.2f}%", "Unknown targets:", f"{ptb_unk_rate * 100:.2f}%")
print()
print("WikiText-2 Test Results")
print("Loss:", f"{wikitext_loss:.4f}")
print("Perplexity:", f"{wikitext_perplexity:.4f}")
print("Accuracy:", f"{wikitext_accuracy * 100:.2f}%")
print("Known-token accuracy:", f"{wiki_known_accuracy * 100:.2f}%", "Unknown targets:", f"{wiki_unk_rate * 100:.2f}%")
