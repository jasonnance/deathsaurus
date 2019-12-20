import enum
import logging
import os
from typing import Tuple

import click
import discord
import torch
import torch.nn as nn
import transformers

from generate import generate_text

GENERATE_LEN = 200

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:S",
    level=logging.INFO,
)

discord_client = discord.Client()

COMMAND_PREFIX = "!xfm"


class Command(enum.Enum):
    HELP = f"{COMMAND_PREFIX}_help"
    GENERATE_TEXT = f"{COMMAND_PREFIX}_gen"
    PING = f"{COMMAND_PREFIX}_ping"


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


USAGE_STR = "\n".join(
    (
        "Deathsaurus is a bot that can perform a few different tasks using Transformer-based deep learning models.",
        "The currently-supported tasks are:",
        f"  {COMMAND_PREFIX}_help: Show this help text.",
        f"  {COMMAND_PREFIX}_ping: Return a response to indicate whether the bot is listening.",
        f"  {COMMAND_PREFIX}_gen <text>: Generate text, using <text> as the starting seed text.",
    )
)


def handle_cmd(
    cmd: Command,
    text: str,
    model: nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
    markdown: bool = False,
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

        if markdown:
            return f"**{text}**{generated_text}"
        else:
            return f"{text}{generated_text}"
    elif cmd == Command.PING:
        return "pong"
    else:
        raise InvalidCommandError(f"Unimplemented command: {cmd}")


def _verify_env(var_name: str, err_msg: str) -> str:
    try:
        return os.environ[var_name]
    except KeyError:
        raise RuntimeError(err_msg)


class DeathsaurusClient(discord.Client):
    def __init__(
        self,
        guild: str,
        channel: str,
        model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        device: torch.device,
        **options,
    ):
        super().__init__(**options)
        self.guild_name = guild
        self.channel_name = channel
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    async def on_ready(self):
        my_guild = None
        for g in self.guilds:
            if g.name == self.guild_name:
                my_guild = g
        if my_guild is None:
            raise RuntimeError(f"Couldn't find guild named {self.guild_name}")
        self.guild = my_guild

        my_channel = None
        for c in self.guild.channels:
            if c.name == self.channel_name:
                my_channel = c
        if my_channel is None:
            raise RuntimeError(
                f"Couldn't find channel named {self.channel_name} in guild {self.guild_name}"
            )
        self.channel = my_channel

        logger.info("Discord bot logged in.")
        await self.channel.send(":robot: Deathsaurus _online_.")

    async def on_message(self, message):
        # Ignore our own messages and messages that aren't in the guild we're watching
        self_message = message.author == self.user
        other_guild_message = message.guild != self.guild
        has_command = message.content.startswith(COMMAND_PREFIX)

        if self_message or other_guild_message or not has_command:
            return

        async with self.channel.typing():
            try:
                cmd, text = parse_cmd(message.content)
                cmd_output = handle_cmd(
                    cmd, text, self.model, self.tokenizer, self.device, markdown=True
                )
            except InvalidCommandError as e:
                cmd_output = f"ERROR: {str(e)}"

            await self.channel.send(cmd_output)

    async def on_disconnect(self):
        await self.channel.send(":wave: Deathsaurus signing off.")


def discord_loop(
    model: nn.Module, tokenizer: transformers.PreTrainedTokenizer, device: torch.device
):
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

    client = DeathsaurusClient(discord_guild, discord_channel, model, tokenizer, device)
    client.run(discord_token)


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
                cmd_output = handle_cmd(
                    cmd, text, model, tokenizer, device, markdown=False
                )
            except InvalidCommandError as e:
                cmd_output = f"ERROR: {str(e)}"
            click.echo(cmd_output)


if __name__ == "__main__":
    run()
