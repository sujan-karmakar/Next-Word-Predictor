# Next Word Predictor

A PyTorch-based next-word prediction system built with a multi-layer GRU language model. The project trains on the text files in `data/train`, evaluates on Penn Treebank and WikiText-2 test sets, and provides an interactive text-generation interface.

## Project Structure

```text
.
├── data/
│   ├── train/                 Training text files
│   ├── validation/            Validation text
│   ├── test/                  PTB and WikiText-2 test text
│   └── processed/             Generated combined training corpus
├── word_mapping/              Generated vocabulary mappings
├── best_model.pth             Saved best model checkpoint
├── preprocessing.py           Cleans and combines training text
├── tokenization.py            Tokenization and vocabulary utilities
├── data_loader.py             Training dataset construction
├── validation_loader.py       Validation dataset construction
├── model.py                   GRU model definition
├── train.py                   Model training
├── evaluate.py                Test-set evaluation
└── predict.py                 Interactive next-word generation
```

## Setup

Use Python 3.9 or newer, then install the dependencies:

```bash
pip install torch regex
```

Run all commands from the project root.

## Usage

### 1. Prepare the training data

This combines and cleans every `.txt` file in `data/train`:

```bash
python preprocessing.py
```

### 2. Train the model

Training creates `word_mapping/word_to_token.json`, `word_mapping/token_to_word.json`, and `best_model.pth`:

```bash
python train.py
```

The best checkpoint is selected by validation loss on `data/validation/ptb_valid.txt`.

### 3. Evaluate the model

Evaluate the saved checkpoint on both supported test datasets:

```bash
python evaluate.py
```

The script reports loss, perplexity, and next-token accuracy for PTB and WikiText-2.

### 4. Generate text interactively

Start the predictor after training:

```bash
python predict.py
```

Enter a prompt and the number of words to generate. Type `exit` at the prompt to stop.

## Model Configuration

The default training configuration is defined in `train.py`:

- Sequence length: 20 tokens (padding is ignored by the GRU)
- Batch size: 64
- Embedding size: 192
- GRU hidden size: 384
- GRU layers: 2
- Dropout: 0.3
- Epochs: 20 (the learning rate is reduced automatically when validation loss stalls)
- Learning rate: 0.001

Prediction uses top-10 sampling with temperature `0.8`. The `<pad>`, `<unk>`, and `<bos>` tokens are excluded from generated output.

## Notes

- Run preprocessing before training if the processed corpus does not exist or the training data changes. It preserves PTB/WikiText `<unk>` markers and treats punctuation as separators, consistently across all splits.
- Run training before evaluation or prediction so the checkpoint and vocabulary mappings match. The scripts deliberately reject a stale checkpoint, rather than producing misleading metrics.
- Test perplexity is the exponential of cross-entropy; it is expected to be much larger than accuracy because it measures the probability assigned to every correct next token. Compare it only across runs using the same tokenization and vocabulary.
- CUDA is used automatically when PyTorch detects a compatible GPU; otherwise, the project runs on CPU.
