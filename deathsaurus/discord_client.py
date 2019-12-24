import functools
import logging

import torch
import torch.nn as nn
import transformers
from deathsaurus.util import COMMAND_PREFIX, Command, InvalidCommandError, handle_cmd
from discord.ext import commands
from discord.utils import find

logger = logging.getLogger(__name__)


class Deathsaurus(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot,
        guild: str,
        channel: str,
        model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        device: torch.device,
    ):
        self.bot = bot
        self.guild_name = guild
        self.channel_name = channel
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @commands.Cog.listener()
    async def on_ready(self):
        my_guild = find(lambda g: g.name == self.guild_name, self.bot.guilds)
        if my_guild is None:
            raise RuntimeError(f"Couldn't find guild named {self.guild_name}")
        self.guild = my_guild

        my_channel = find(lambda c: c.name == self.channel_name, self.guild.channels)
        if my_channel is None:
            raise RuntimeError(
                f"Couldn't find channel named {self.channel_name} in guild {self.guild_name}"
            )
        self.channel = my_channel

        logger.info("Discord bot logged in.")
        await self.channel.send(":robot: Deathsaurus _online_.")

    def _handle_cmd(self, cmd: Command, text: str) -> str:
        return handle_cmd(
            cmd, text, self.model, self.tokenizer, self.device, markdown=True
        )

    @commands.command(
        help="Return a response to indicate whether the bot is listening."
    )
    @commands.guild_only()
    async def ping(self, ctx):
        await self.channel.send(self._handle_cmd(Command.PING, ""))

    @commands.command(help="Generate text from the given seed text.")
    @commands.guild_only()
    async def gen(self, ctx, *, seed_text: str):
        async with ctx.typing():
            try:
                # Run the generation asynchronously so our Discord connection stays active
                generate_text = functools.partial(
                    self._handle_cmd, Command.GENERATE_TEXT, seed_text
                )
                output = await self.bot.loop.run_in_executor(None, generate_text)
                await self.channel.send(output)
            except InvalidCommandError as e:
                raise commands.BadArgument(str(e))

    async def say_bye(self):
        await self.channel.send(":wave: Deathsaurus signing off.")

    @commands.Cog.listener()
    async def on_disconnect(self):
        await self.say_bye()


def get_bot(
    guild: str,
    channel: str,
    model: nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
) -> commands.Bot:
    """
    Initialize and return a Discord bot that can listen to and respond to
    commands.

    Args:
      guild: Discord server to listen on.
      channel: Discord channel to post output in.
      model: Transformers model to use for evaluating commands.
      tokenizer: Tokenizer to use for parsing input text.
      device: Device the model is on.

    Returns:
      The initialized bot.
    """
    bot = commands.Bot(
        command_prefix=commands.when_mentioned_or(COMMAND_PREFIX),
        description="Bot for interfacing with Transformer neural network models for whacky hijinks.",
    )
    bot.add_cog(Deathsaurus(bot, guild, channel, model, tokenizer, device))
    return bot
