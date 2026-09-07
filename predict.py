import torch
from preprocessing import preprocess_text
from tokenization import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, load_mappings, vocabulary_fingerprint
from model import GRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20
MODEL_PATH = "best_model.pth"
TOP_K = 10
TEMPERATURE = 0.8

word_to_token, token_to_word = load_mappings("word_mapping")
vocab_size = len(word_to_token)
pad_token = word_to_token[PAD_TOKEN]
unk_token = word_to_token[UNK_TOKEN]

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if checkpoint.get("vocabulary_fingerprint") != vocabulary_fingerprint(word_to_token):
    raise RuntimeError("Checkpoint and word_mapping do not match. Run preprocessing.py and train.py together before prediction.")

model = GRU(vocab_size=vocab_size, embedding_dim=checkpoint["embedding_dim"], hidden_size=checkpoint["hidden_size"], num_layers=checkpoint["num_layers"], dropout=checkpoint["dropout"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def predict_next_word(tokens):
    tokens = tokens[-SEQ_LEN:]
    length = len(tokens)
    tokens = tokens + [pad_token] * (SEQ_LEN - length)
    X = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([length], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(X, lengths)
        logits[:, pad_token] = float("-inf")
        logits[:, unk_token] = float("-inf")
        logits[:, word_to_token[BOS_TOKEN]] = float("-inf")
        logits = logits / TEMPERATURE
        top_logits, top_tokens = torch.topk(logits, TOP_K, dim=1)
        probabilities = torch.softmax(top_logits, dim=1)
        selected_index = torch.multinomial(probabilities, 1).item()
        next_token = top_tokens[0, selected_index].item()

    return next_token

def generate_text(text, num_words):
    words = preprocess_text(text).split()
    tokens = [word_to_token[BOS_TOKEN]] + [word_to_token.get(word, unk_token) for word in words]

    generated_words = []

    for _ in range(num_words):
        next_token = predict_next_word(tokens)
        if next_token == word_to_token[EOS_TOKEN]:
            break
        next_word = token_to_word[next_token]
        tokens.append(next_token)
        generated_words.append(next_word)

    return " ".join(words + generated_words)

print("Device:", DEVICE)
print("Vocabulary size:", vocab_size)
print("Model loaded from:", MODEL_PATH)
print("Top-K:", TOP_K)
print("Temperature:", TEMPERATURE)
print()

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        print("Prediction stopped.")
        break

    if not text.strip():
        print("Please enter some text.")
        continue

    num_words = input("Number of words to generate: ")

    try:
        num_words = int(num_words)
        if num_words <= 0:
            print("Enter a positive number.")
            print()
            continue
    except ValueError:
        print("Please enter a valid number.")
        print()
        continue

    generated_text = generate_text(text, num_words)

    print()
    print("Generated text:")
    print(generated_text)
    print()
