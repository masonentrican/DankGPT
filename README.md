# DankGPT

A GPT-style language model in progress, built with Python, PyTorch, and tiktoken.

## Setup

```bash
git clone https://github.com/masonentrican/DankGPT.git
cd DankGPT

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -e .
```

## Prepare Data

```bash
python scripts/prepare_data.py
```

Downloads example text into `data/raw/`.

## Train Model

```bash
python scripts/train.py
```

Runs a small GPT-style model on the prepared dataset.

## Project Structure

```text
src/llm/     # library code (dataset, tokenizer, model, training)
scripts/     # run scripts (prepare_data, train)
data/raw/    # downloaded data (ignored in git)
configs/     # optional configs
```

## Notes

- `data/raw/`, `.venv/`, and build artifacts are ignored by git
- Installed in editable mode via `pip install -e .`
