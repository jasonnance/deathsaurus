import enum
import logging
from typing import Tuple

import click
import torch
import torch.nn as nn
import transformers

from generate import generate_text

GENERATE_LEN = 400

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:S",
    level=logging.INFO,
)


class Command(enum.Enum):
    HELP = "!xfm_help"
    GENERATE_TEXT = "!xfm_gen"


class InvalidCommandError(Exception):
    pass


def parse_cmd(cmd_str: str) -> Tuple[Command, str]:
    first_whitespace_ndx = cmd_str.find(" ")
    if first_whitespace_ndx == -1:
        cmd_prefix = cmd_str
        cmd_text = ""
    else:
        cmd_prefix = cmd_str[:first_whitespace_ndx]
        cmd_text = cmd_str[first_whitespace_ndx:].strip()

    try:
        cmd = Command(cmd_prefix)
    except ValueError:
        raise InvalidCommandError(f"Unknown command name: '{cmd_prefix}'")

    return cmd, cmd_text


def discord_loop(
    model: nn.Module, tokenizer: transformers.PreTrainedTokenizer, device: torch.device
):
    raise NotImplementedError


USAGE_STR = "\n".join(
    (
        "Deathsaurus is a bot that can perform a few different tasks using Transformer-based deep learning models.",
        "The currently-supported tasks are:",
        "  !xfm_help: Show this help text.",
        "  !xfm_gen <text>: Generate text, using <text> as the starting seed text.",
    )
)


def handle_cmd(
    cmd: Command,
    text: str,
    model: nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
) -> str:
    if cmd == Command.HELP:
        return USAGE_STR
    elif cmd == Command.GENERATE_TEXT:
        if len(text) == 0:
            raise InvalidCommandError("Text generation requires nonempty seed text.")
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            seed_text=text,
            device=device,
            length=GENERATE_LEN,
        )

        return "\n".join((text, generated_text))
    else:
        raise InvalidCommandError(f"Unimplemented command: {cmd}")


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
@click.option(
    "--run-discord/--run-local",
    help="Whether to run the bot accepting input from a Discord server or "
    "locally on the terminal.",
    default=False,
)
def run(model_name: str, cuda: bool, cache_dir: str, run_discord: bool):
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

    if run_discord:
        discord_loop(model, tokenizer, device)
    else:
        while True:
            cmd_str = input("> ")
            try:
                cmd, text = parse_cmd(cmd_str)
                cmd_output = handle_cmd(cmd, text, model, tokenizer, device)
            except InvalidCommandError as e:
                cmd_output = f"ERROR: {str(e)}"
            click.echo(cmd_output)


if __name__ == "__main__":
    run()
