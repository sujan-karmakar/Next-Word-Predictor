from pathlib import Path
import regex as re

def preprocess_text(text):
    text = re.sub(r"[^\p{L}\s]", "", text) # Remove punctuation and special characters

    text = re.sub(r"[ \t]+", " ", text) # Remove extra spaces/tabs

    text = re.sub(r"\n+", "\n", text).strip() # Remove unnecessary blank lines

    text = re.sub(r"<.*?>", "", text).strip() # Remove angular brackets

    return text


folder_path = Path("data/train")
combined_text = ""

for file_path in folder_path.glob("*.txt"):
    combined_text += file_path.read_text(encoding="utf-8") + "\n"

combined_text = preprocess_text(combined_text)

output_path = Path("data/processed/combined_text.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(combined_text, encoding="utf-8")
