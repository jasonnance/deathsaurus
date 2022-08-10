import logging
import os

import click
import discord
import torch
import torch.nn as nn
import transformers

from deathsaurus.discord_client import get_bot
from deathsaurus.repl_client import repl

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:S",
    level=logging.INFO,
)

discord_client = discord.Client()


def _verify_env(var_name: str, err_msg: str) -> str:
    """
    Verify the given environment variable is set and return its value.
    Raise an error if it isn't set.

    Args:
      var_name: Environment variable name
      err_msg: Error message to show if the variable isn't set

    Returns:
      The value of the environment variable
    """
    try:
        return os.environ[var_name]
    except KeyError:
        raise RuntimeError(err_msg)


def discord_loop(
    model: nn.Module,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    device: torch.device,
):
    """
    Run the async Discord bot loop.

    Args:
      model: Model to use for handling commands.
      tokenizer: Tokenizer for parsing input and decoding output.
      device: Device the model is sitting on.
    """
    discord_token = _verify_env(
        "DISCORD_BOT_TOKEN",
        "Environment variable DISCORD_BOT_TOKEN must be set to the bot token for login.",
    )
    discord_guild = _verify_env(
        "DISCORD_BOT_GUILD",
        "Environment variable DISCORD_BOT_GUILD must be set to the server name to listen on.",
    )
    discord_channel = _verify_env(
        "DISCORD_BOT_CHANNEL",
        "Environment variable DISCORD_BOT_CHANNEL must be set to the channel name to post in.",
    )
    bot = get_bot(discord_guild, discord_channel, model, tokenizer, device)
    bot.run(discord_token)


@click.command()
@click.option(
    "--model-name",
    help="Transformer model weights to use.  The model must "
    "be the name of one of the pretrained models supported by the transformers "
    "library or a path to a custom weights file.",
    default="gpt2-large",
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
@click.option(
    "--download-only/--no-download-only",
    help="If True, only download the model files to the cache directory and exit.",
    default=False,
)
def main(
    model_name: str, cuda: bool, cache_dir: str, run_discord: bool, download_only: bool
):
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

    if download_only:
        logger.info(f"Ensured model '{model_name}' is downloaded to cache directory.")
        return

    model.to(device)
    model.eval()
    logger.info(f"Loaded model and tokenizer: '{model_name}'")

    if run_discord:
        discord_loop(model, tokenizer, device)
    else:
        repl(model, tokenizer, device)


if __name__ == "__main__":
    main()
