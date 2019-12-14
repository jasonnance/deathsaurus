import logging

import click
import torch
import transformers

from generate import generate_text

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:S",
    level=logging.INFO,
)


@click.command()
@click.option(
    "--model-name",
    help="Transformer model weights to use.  The model must "
    "be the name of one of the pretrained models supported by the transformers "
    "library or a path to a custom weights file.",
    default="gpt2",
    show_default=True,
)
@click.option(
    "--cuda/--no-cuda",
    help="Whether to use GPU.  Defaults to GPU if available.",
    default=True,
)
@click.option(
    "--cache-dir",
    help="Directory to cache transformers weights downloads.",
    default="/cache",
    type=click.Path(exists=True, file_okay=False, writable=True),
    show_default=True,
)
def run(model_name: str, cuda: bool, cache_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device} ({n_gpu} GPUs)")

    logger.info("Loading model and tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    model = transformers.AutoModelWithLMHead.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    model.to(device)
    model.eval()
    logger.info(f"Loaded model and tokenizer: '{model_name}'")

    seed_text = "Asad thought I wouldn't do it, but he was woefully mistaken."

    generated_text = generate_text(
        model=model, tokenizer=tokenizer, seed_text=seed_text, device=device, length=400
    )

    print(seed_text)
    print(generated_text)


if __name__ == "__main__":
    run()
