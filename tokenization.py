from pathlib import Path
import json

file_path = Path("data/processed/combined_text.txt")
text = file_path.read_text(encoding="utf-8")

lines = text.splitlines()

# Tokenise words
tokenized_lines = [line.split() for line in lines]

# Create word -> token
word_to_token = {}
token = 1

for line in tokenized_lines:
    for word in line:
        if word not in word_to_token:
            word_to_token[word] = token
            token += 1

# Create token -> word
token_to_word = {
    token: word
    for word, token in word_to_token.items()
}

# Vectorize
vectorized_lines = [
    [word_to_token[word] for word in line]
    for line in tokenized_lines
]

with open("word mapping/word_to_token.json", "w", encoding="utf-8") as f:
    json.dump(word_to_token, f, ensure_ascii=False, indent=2)

with open("word mapping/token_to_word.json", "w", encoding="utf-8") as f:
    json.dump(token_to_word, f, ensure_ascii=False, indent=2)