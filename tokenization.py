from pathlib import Path
from collections import Counter
import hashlib
import json
from preprocessing import preprocess_text


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


def tokenize_file(file_path):
    text = Path(file_path).read_text(encoding="utf-8")
    text = preprocess_text(text)
    lines = text.splitlines()
    return [line.split() for line in lines if line.strip()]


def create_vocabulary(tokenized_lines, min_frequency=2, max_vocabulary_size=30000):
    word_counts = Counter(word for line in tokenized_lines for word in line)

    ranked_words = sorted(
        (word for word, count in word_counts.items() if count >= min_frequency and word not in SPECIAL_TOKENS),
        key=lambda word: (-word_counts[word], word),
    )
    if max_vocabulary_size is not None:
        ranked_words = ranked_words[:max(0, max_vocabulary_size - len(SPECIAL_TOKENS))]
    word_to_token = {word: index for index, word in enumerate(list(SPECIAL_TOKENS) + ranked_words)}

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


def vocabulary_fingerprint(word_to_token):
    payload = json.dumps(word_to_token, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
