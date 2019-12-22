import click
import torch
import torch.nn as nn
import transformers

from util import InvalidCommandError, handle_cmd, parse_cmd


def repl(
    model: nn.Module, tokenizer: transformers.PreTrainedTokenizer, device: torch.device
):
    """
    Run a read-eval-print loop through the command line.

    Args:
      model: Transformers model to use for evaluating commands.
      tokenizer: Tokenizer to use for parsing input text.
      device: Device the model is on.
    """
    click.echo("Type 'help' to see info on available commands.")
    while True:
        cmd_str = input("> ")
        try:
            cmd, text = parse_cmd(cmd_str)
            cmd_output = handle_cmd(cmd, text, model, tokenizer, device, markdown=False)
        except InvalidCommandError as e:
            cmd_output = f"ERROR: {str(e)}"
        click.echo(cmd_output)
