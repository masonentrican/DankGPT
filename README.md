# DankGPT

A minimal GPT-style language model project using Python, PyTorch, and tiktoken.  
Includes text downloading, tokenization, dataset windowing, and a simple training loop.

---

## Setup

git clone https://github.com/masonentrican/DankGPT.git  
cd DankGPT

python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows

pip install -e .

---

## Prepare Data

python scripts/prepare_data.py

Downloads example text into data/raw/.

---

## Train Model

python scripts/train.py

Runs a small GPT-style model on the prepared dataset.

---

## Project Structure

src/llm/        # library code (dataset, tokenizer, model, training)  
scripts/        # run scripts (prepare_data, train)  
data/raw/       # downloaded data (ignored in git)  
configs/        # optional configs

---

## Notes

- data/raw/, .venv/, and build artifacts are ignored by git  
- Project is installed in editable mode via: pip install -e .

