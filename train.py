import torch
import torch.nn as nn
from torch.optim import Adam
from data_loader import create_train_loader
from validation_loader import create_validation_loader
from model import GRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2

train_loader, word_to_token, token_to_word = create_train_loader()
val_loader = create_validation_loader()

vocab_size = len(word_to_token)

model = GRU(
    vocab_size=vocab_size, 
    embedding_dim=EMBEDDING_DIM, 
    hidden_size=HIDDEN_SIZE, 
    num_layers=NUM_LAYERS, 
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")

print("Device:", DEVICE)
print("Vocabulary size:", vocab_size)
print("Total parameters:", sum(p.numel() for p in model.parameters()))

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(val_loader)

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": average_train_loss,
            "val_loss": average_val_loss,
            "vocab_size": vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT
        }, "best_model.pth")

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {average_train_loss:.4f} Val Loss: {average_val_loss:.4f} ✓ Best model saved")
    else:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {average_train_loss:.4f} Val Loss: {average_val_loss:.4f}")

print("Training completed.")
print("Best validation loss:", best_val_loss)
print("Best model saved as best_next_word_gru.pth")