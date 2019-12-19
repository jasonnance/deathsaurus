"""
Adapted from:
https://github.com/huggingface/transformers/blob/e92bcb7eb6c5b9b6ed313cc74abaab50b3dc674f/examples/run_generation.py
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import trange

MAX_LENGTH = 1000

logger = logging.getLogger(__name__)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def generate_text(
    *,
    model: nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seed_text: str,
    device: torch.device,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    length: int = -1,
    num_samples: int = 1,
) -> str:
    """
    Generate text using the given model and seed text.

    Args:
      model: Model to use for generation.
      seed_text: Text to seed the text generation process.
      device: Torch device containing the model
      temperature: Closer to 0 means more variation in the generated text, while
        closer to 1 means less variation
      top_k: If greater than 0, keep only top k tokens with highest probability
      top_p: If greater than 0, keep top tokens with cumulative probability >= top_p.
      repetition_penalty: Penalize the model when making repeated predictions if this is
        greater than 1.
      length: If greater than 0, limit generation to this many tokens.
      num_samples: Number of samples to generate.
    """
    context_tokens = tokenizer.encode(seed_text, add_special_tokens=False)
    context_len = len(context_tokens)

    generate_length = model.config.max_position_embeddings - context_len  # type: ignore
    if length > 0:
        generate_length = min(length, generate_length)
    elif generate_length < 0:
        # Avoid an infinite generation loop
        generate_length = MAX_LENGTH

    logger.info(
        f"Generating {generate_length} tokens from context of length {context_len}"
    )

    generated = (
        torch.tensor(context_tokens, dtype=torch.long, device=device)
        .unsqueeze(0)
        .repeat(num_samples, 1)
    )
    with torch.no_grad():
        for _ in trange(generate_length):
            inputs = {"input_ids": generated}
            outputs = model(**inputs)

            # Calculate the logits
            next_token_logits = outputs[0][:, -1, :]
            if temperature > 0:
                next_token_logits /= temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for sample_ndx in range(num_samples):
                    for generated_ndx in generated[sample_ndx].tolist():
                        next_token_logits[
                            sample_ndx, generated_ndx
                        ] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            if temperature == 0:  # greedy sampling
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1
                )
            generated = torch.cat((generated, next_token), dim=1)

    generated_only = generated[:, context_len:].tolist()
    text = []
    for generated_token in generated_only:
        text.append(
            tokenizer.decode(generated_token, clean_up_tokenization_spaces=True)
        )
    return "".join(text)
