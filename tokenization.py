from pathlib import Path
from collections import Counter
import json
from preprocessing import preprocess_text


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize_file(file_path):
    text = Path(file_path).read_text(encoding="utf-8")
    text = preprocess_text(text)
    lines = text.splitlines()
    return [line.split() for line in lines if line.strip()]


def create_vocabulary(tokenized_lines):
    word_counts = Counter(word for line in tokenized_lines for word in line)

    word_to_token = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    token = 2

    for word in word_counts:
        word_to_token[word] = token
        token += 1

    token_to_word = {token: word for word, token in word_to_token.items()}

    return word_to_token, token_to_word


def vectorize_lines(tokenized_lines, word_to_token):
    unk_token = word_to_token[UNK_TOKEN]

    vectorized_lines = [
        [word_to_token.get(word, unk_token) for word in line]
        for line in tokenized_lines
    ]

    return vectorized_lines


def save_mappings(word_to_token, token_to_word, folder_path="word_mapping"):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "word_to_token.json", "w", encoding="utf-8") as f:
        json.dump(word_to_token, f, ensure_ascii=False, indent=2)

    with open(folder / "token_to_word.json", "w", encoding="utf-8") as f:
        json.dump(token_to_word, f, ensure_ascii=False, indent=2)


def load_mappings(folder_path="word_mapping"):
    folder = Path(folder_path)

    with open(folder / "word_to_token.json", "r", encoding="utf-8") as f:
        word_to_token = json.load(f)

    with open(folder / "token_to_word.json", "r", encoding="utf-8") as f:
        token_to_word = json.load(f)

    token_to_word = {int(token): word for token, word in token_to_word.items()}

    return word_to_token, token_to_word