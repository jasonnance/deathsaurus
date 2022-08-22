import os

import click
import discord
import torch
import torch.nn as nn
import transformers
from loguru import logger

from deathsaurus import generate_image_stable_diffusion
from deathsaurus.discord_client import get_image_bot, get_text_bot
from deathsaurus.repl_client import image_repl, text_repl

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


def get_hf_env():
    hf_hub_token = _verify_env(
        "HF_HUB_TOKEN",
        "Environment variable HF_HUB_TOKEN must be set to a valid token for accessing "
        "HuggingFace Hub.",
    )
    return hf_hub_token


def get_discord_env():
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
    discord_hof_channel = _verify_env(
        "DISCORD_BOT_HOF_CHANNEL",
        "Environment variable DISCORD_BOT_HOF_CHANNEL must be set to the channel name to "
        "use for the Hall of Fame.",
    )
    return (
        discord_token,
        discord_guild,
        discord_channel,
        discord_hof_channel,
    )


def text_discord_loop(
    model: nn.Module,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    device: torch.device,
):
    """
    Run the async Discord bot loop for text.

    Args:
      model: Model to use for handling commands.
      tokenizer: Tokenizer for parsing input and decoding output.
      device: Device the model is sitting on.
    """
    (
        discord_token,
        discord_guild,
        discord_channel,
        discord_hof_channel,
    ) = get_discord_env()
    bot = get_text_bot(
        discord_guild, discord_channel, discord_hof_channel, model, tokenizer, device
    )
    bot.run(discord_token)


def run_text(
    model_name: str, cuda: bool, cache_dir: str, download_only: bool, run_discord: bool
):
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device} ({n_gpu} GPUs)")

    logger.info("Loading model and tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(  # type: ignore
        model_name, cache_dir=cache_dir
    )
    model = transformers.AutoModelWithLMHead.from_pretrained(  # type: ignore
        model_name, cache_dir=cache_dir
    )

    if download_only:
        logger.info(f"Ensured model '{model_name}' is downloaded to cache directory.")
        return

    model.to(device)
    model.eval()
    logger.info(f"Loaded model and tokenizer: '{model_name}'")

    if run_discord:
        text_discord_loop(model, tokenizer, device)
    else:
        text_repl(model, tokenizer, device)


def image_discord_loop(pipe):
    """
    Run the async Discord bot loop for images.
    """
    (
        discord_token,
        discord_guild,
        discord_channel,
        discord_hof_channel,
    ) = get_discord_env()
    bot = get_image_bot(
        discord_guild,
        discord_channel,
        discord_hof_channel,
        pipe,
    )
    bot.run(discord_token)


def run_image(run_discord: bool, cuda: bool):
    hf_hub_token = get_hf_env()
    pipe = generate_image_stable_diffusion.init_model(hf_hub_token, cuda)

    if run_discord:
        image_discord_loop(pipe)
    else:
        image_repl(pipe)


@click.command()
@click.option(
    "--mode",
    help="What to generate.  text or image",
    required=True,
    type=click.Choice(["text", "image"]),
)
@click.option(
    "--text-model-name",
    help="Transformer model weights to use.  The model must "
    "be the name of one of the pretrained models supported by the transformers "
    "library or a path to a custom weights file.  Ignored if mode = image.",
    default="gpt2-large",
    show_default=True,
)
@click.option(
    "--cuda/--no-cuda",
    help="Whether to use GPU.  Defaults to GPU if available. Ignored if mode = image.",
    default=True,
)
@click.option(
    "--cache-dir",
    help="Directory to cache transformers weights downloads. Ignored if mode = image.",
    default="/tmp",
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
    help="If True, only download the model files to the cache directory and exit. Ignored if mode = image.",
    default=False,
)
def main(
    mode: str,
    text_model_name: str,
    cuda: bool,
    cache_dir: str,
    run_discord: bool,
    download_only: bool,
):
    if mode == "text":
        run_text(text_model_name, cuda, cache_dir, download_only, run_discord)
    elif mode == "image":
        run_image(run_discord, cuda)
    else:
        raise ValueError(f"unsupported mode: {mode}")


if __name__ == "__main__":
    main()
