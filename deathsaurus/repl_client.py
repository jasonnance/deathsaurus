import typing
import uuid
from io import BytesIO
from pathlib import Path

import click
import torch
import torch.nn as nn
import transformers
from PIL import Image

from deathsaurus.util import (
    Command,
    InvalidCommandError,
    handle_cmd_image,
    handle_cmd_text,
    parse_cmd,
)

# Path to save generated images from the REPL in
IMG_PATH = Path(__file__).parent.parent / "img"


def _repl(handle_func: typing.Callable[[Command, str], str]):
    """
    Base function for running REPLs.
    """
    click.echo("Type 'help' to see info on available commands.")
    while True:
        cmd_str = input("> ")
        try:
            cmd, text = parse_cmd(cmd_str)
            cmd_output = handle_func(
                cmd,
                text,
            )
        except InvalidCommandError as e:
            cmd_output = f"ERROR: {str(e)}"
        click.echo(cmd_output)


def text_repl(
    model: nn.Module,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    device: torch.device,
):
    """
    Run a read-eval-print loop through the command line.

    Args:
      model: Transformers model to use for evaluating commands.
      tokenizer: Tokenizer to use for parsing input text.
      device: Device the model is on.
    """

    def handle_func(cmd, text):
        return handle_cmd_text(cmd, text, model, tokenizer, device, markdown=False)

    return _repl(handle_func)


def image_repl(pipe):
    """
    Run a read-eval-print loop through the command line.  Clears out existing
    images.
    """
    IMG_PATH.mkdir(exist_ok=True, parents=True)
    for existing_img in IMG_PATH.iterdir():
        if existing_img.suffix == ".png":
            existing_img.unlink()

    def file_callback(file: BytesIO):
        img = Image.open(file)
        filename = IMG_PATH / f"{uuid.uuid4().hex}.png"
        img.save(filename)
        print(f"Image saved to: {filename}")

    def handle_func(cmd, text):
        result = handle_cmd_image(cmd, text, pipe)
        if isinstance(result, str):
            return result
        elif isinstance(result, list):
            for img in result:
                file_callback(img)
            return "Images saved."
        else:
            raise TypeError(f"Unexpected output type: {type(result)}")

    return _repl(handle_func)
