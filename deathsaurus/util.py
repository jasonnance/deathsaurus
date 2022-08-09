import enum
from typing import Tuple

import torch
import torch.nn as nn
import transformers
from discord.utils import escape_markdown

from deathsaurus.generate import generate_text

COMMAND_PREFIX = "!xfm_"
GENERATE_LEN = 200


class Command(enum.Enum):
    HELP = "help"
    GENERATE_TEXT = "gen"
    PING = "ping"


class InvalidCommandError(Exception):
    pass


def parse_cmd(cmd_str: str) -> Tuple[Command, str]:
    """
    Parse a bot command from the given command string.  The string is assumed to have
    had the command prefix already stripped from it.

    Args:
      cmd_str: String containing both a command and whatever text follows it.

    Returns:
      2-tuple: the type of command and any text associated with it.
    """
    first_whitespace_ndx = cmd_str.find(" ")
    if first_whitespace_ndx == -1:
        cmd_name = cmd_str
        cmd_text = ""
    else:
        cmd_name = cmd_str[:first_whitespace_ndx]
        cmd_text = cmd_str[first_whitespace_ndx:].strip()

    try:
        cmd = Command(cmd_name)
    except ValueError:
        raise InvalidCommandError(f"Unknown command name: '{cmd_name}'")

    return cmd, cmd_text


USAGE_STR = "\n".join(
    (
        "Deathsaurus is a bot that can perform a few different tasks using Transformer-based deep learning models.",
        "The currently-supported tasks are:",
        "  help: Show this help text.",
        "  ping: Return a response to indicate whether the bot is listening.",
        "  gen <text>: Generate text, using <text> as the starting seed text.",
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
    """
    Handle the given command and return the generated output.

    Args:
      cmd: Type of command to handle.
      text: Extra text for the command, if applicable.
      model: Model to use for the command.
      tokenizer: Tokenizer to use for parsing the text, if needed.
      device: Device the model is currently on.
      markdown: If True, format results assuming they'll be displayed in Markdown.
        Otherwise, assume raw text output

    Returns:
      Generated output text.
    """
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
            return f"**{text}**{escape_markdown(generated_text)}"
        else:
            return f"{text}{generated_text}"
    elif cmd == Command.PING:
        return "pong"
    else:
        raise InvalidCommandError(f"Unimplemented command: {cmd}")
