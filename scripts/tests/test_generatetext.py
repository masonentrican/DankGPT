import torch
from llm.generation.generator import generate_text
from llm.utils import get_tokenizer
from llm.utils.tokenization import text_to_token_ids, token_ids_to_text
from config.models import SMOOTHBRAIN, GPT2_SMALL
from llm.models.gptmodel import GPTModel

def main():
    """
    Main function to test the text generation functionality with temperature scaling and top-k sampling.
    """
    torch.manual_seed(123)
    model = GPTModel(GPT2_SMALL)
    model.to("cpu")
    model.eval()
    tokenizer = get_tokenizer()

    token_ids = generate_text(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT2_SMALL["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    main()