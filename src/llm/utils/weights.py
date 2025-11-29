"""
Weight initialization utilities.
"""

import torch
import numpy as np


def assign(left: torch.Tensor, right: torch.Tensor) -> torch.nn.Parameter:
    """
    Assign a tensor to a parameter, checking for shape mismatch.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_openai_weights_into_gpt(gpt, params):
    """
    Load OpenAI weights into GPT model.

    Args:
        gpt: GPT model to load weights into.
        params: Dictionary of weights to load.

    Returns:
        None
    """
    # Positional and token embedding weights
    gpt.emb.pos_emb.weight = assign(gpt.emb.pos_emb.weight, params['wpe'])
    gpt.emb.tok_emb.weight = assign(gpt.emb.tok_emb.weight, params['wte'])

    # Transformer blocks
    for b in range(len(params["blocks"])):
        # Attention weights
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].attention.weight_query.weight = assign(
            gpt.trf_blocks[b].attention.weight_query.weight, q_w.T
        )
        gpt.trf_blocks[b].attention.weight_key.weight = assign(
            gpt.trf_blocks[b].attention.weight_key.weight, k_w.T
        )
        gpt.trf_blocks[b].attention.weight_value.weight = assign(
            gpt.trf_blocks[b].attention.weight_value.weight, v_w.T
        )

        # Attention biases
        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].attention.weight_query.bias = assign(
            gpt.trf_blocks[b].attention.weight_query.bias, q_b
        )
        gpt.trf_blocks[b].attention.weight_key.bias = assign(
            gpt.trf_blocks[b].attention.weight_key.bias, k_b
        )
        gpt.trf_blocks[b].attention.weight_value.bias = assign(
            gpt.trf_blocks[b].attention.weight_value.bias, v_b
        )

        # Attention output projection
        gpt.trf_blocks[b].attention.out_projection.weight = assign(
            gpt.trf_blocks[b].attention.out_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].attention.out_projection.bias = assign(
            gpt.trf_blocks[b].attention.out_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # Feed-forward layers
        gpt.trf_blocks[b].feedforward.layers[0].weight = assign(
            gpt.trf_blocks[b].feedforward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].feedforward.layers[0].bias = assign(
            gpt.trf_blocks[b].feedforward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].feedforward.layers[2].weight = assign(
            gpt.trf_blocks[b].feedforward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].feedforward.layers[2].bias = assign(
            gpt.trf_blocks[b].feedforward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        # Layer norms
        gpt.trf_blocks[b].normalization1.scale = assign(
            gpt.trf_blocks[b].normalization1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].normalization1.shift = assign(
            gpt.trf_blocks[b].normalization1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].normalization2.scale = assign(
            gpt.trf_blocks[b].normalization2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].normalization2.shift = assign(
            gpt.trf_blocks[b].normalization2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

    # Final layer norm and output head
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # OpenAI GPT-2 model re-uses the token embedding weights for the output head (weight tying)
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])