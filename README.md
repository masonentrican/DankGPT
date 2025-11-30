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

## Download OpenAI Weights (Optional)

```bash
python scripts/load_gpt2_model.py
```

Downloads GPT-2 124M weights from OpenAI into `models/gpt2/`.

## Train Model

```bash
python scripts/tests/test_train.py
```

Trains a small GPT-style model on the prepared dataset with checkpoint saving/loading.

## Testing

Run the full test suite:

```bash
python scripts/perform_tests.py
```

This executes all tests in `scripts/tests/` and provides a summary. Available test categories:

- **Model Components**: Attention mechanisms (causal, multi-head, self-attention), transformer blocks, normalization layers, GPT model architecture
- **Activation Functions**: GELU vs ReLU comparisons
- **Text Generation**: Generation with temperature scaling and top-k sampling
- **Training**: Full training loop, loss calculation, checkpoint saving/loading
- **Integration**: Loading OpenAI GPT-2 weights, model size validation
- **Configuration**: Model configuration validation

Individual tests can be run directly, e.g., `python scripts/tests/test_gptmodel.py`.

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
