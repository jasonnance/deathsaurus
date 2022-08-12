import functools
import io
import typing

import discord
import torch
import torch.nn as nn
import transformers
from discord.ext import commands
from discord.utils import find
from loguru import logger

from deathsaurus.util import (
    COMMAND_PREFIX,
    Command,
    InvalidCommandError,
    handle_cmd_image,
    handle_cmd_text,
)


class Deathsaurus(commands.Cog):
    """
    Base class for Discord functionality
    """

    HOF_ORIGINAL_MESSAGE_ID_FIELD_NAME = "Original Message ID"

    def __init__(
        self,
        bot: commands.Bot,
        guild: str,
        channel: str,
        hof_channel: str,
    ):
        self.bot = bot
        self.guild_name = guild
        self.channel_name = channel
        self.hof_channel_name = hof_channel

    @property
    def name(self):
        return "Deathsaurus"

    def _handle_cmd(self, cmd: Command, text: str) -> typing.Any:
        raise NotImplementedError

    @commands.command(help="Describes how to use the Hall of Fame.")
    async def hof(self, ctx):
        await self.channel.send(self._handle_cmd(Command.HALL_OF_FAME, ""))

    @commands.command(
        help="Return a response to indicate whether the bot is listening."
    )
    @commands.guild_only()
    async def ping(self, ctx):
        await self.channel.send(self._handle_cmd(Command.PING, ""))

    @commands.Cog.listener()
    async def on_command_error(self, ctx, exception):
        await self.channel.send(str(exception))

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

        hof_channel = find(
            lambda c: c.name == self.hof_channel_name, self.guild.channels
        )
        if hof_channel is None:
            raise RuntimeError(
                f"Couldn't find channel named {self.hof_channel_name} in guild {self.guild_name}"
            )
        self.hof_channel = hof_channel

        logger.info("Loading existing Hall of Fame messages.")
        self.hof_message_ids = set()
        if self.bot.user is None:
            raise RuntimeError("Bot user isn't logged in.")

        async for message in self.hof_channel.history(limit=None):
            if message.author.id != self.bot.user.id:
                continue
            for embed in message.embeds:
                if embed.footer != discord.Embed.Empty:
                    try:
                        self.hof_message_ids.add(int(embed.footer.text))
                    except TypeError:
                        logger.warning(f"Embed footer had an invalid int: {embed}")
        logger.info(
            f"Loaded {len(self.hof_message_ids)} existing Hall of Fame messages."
        )

        logger.info("Discord bot logged in.")
        await self.channel.send(f":robot: {self.name} _online_.")

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload):
        if self.bot.user is None:
            logger.warning("Ignoring reaction -- not logged in")
            return
        message_id = payload.message_id
        if (
            payload.channel_id == self.channel.id
            and payload.event_type == "REACTION_ADD"
            and str(payload.emoji) == "🏆"
            and message_id not in self.hof_message_ids
        ):
            message = await self.channel.fetch_message(message_id)
            if message.author.id == self.bot.user.id:
                logger.info(f"Adding message ID {message_id} to Hall of Fame")
                embed = discord.Embed()
                embed.set_footer(text=message_id)
                await self.hof_channel.send(
                    message.content,
                    files=[
                        await attachment.to_file() for attachment in message.attachments
                    ],
                    embed=embed,
                )
                self.hof_message_ids.add(message_id)

    async def say_bye(self):
        await self.channel.send(f":wave: {self.name} signing off.")

    @commands.Cog.listener()
    async def on_disconnect(self):
        await self.say_bye()


class DeathsaurusText(Deathsaurus):
    def __init__(
        self,
        bot: commands.Bot,
        guild: str,
        channel: str,
        hof_channel: str,
        model: nn.Module,
        tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
        device: torch.device,
    ):
        super().__init__(bot, guild, channel, hof_channel)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.running_gen = False

    @property
    def name(self):
        return "Deathsaurus for Text Generation"

    def _handle_cmd(self, cmd: Command, text: str) -> str:
        return handle_cmd_text(
            cmd, text, self.model, self.tokenizer, self.device, markdown=True
        )

    @commands.command(help="Generate text from the given seed text.")
    @commands.guild_only()
    async def gen(self, ctx, *, seed_text: str):
        if self.running_gen:
            await self.channel.send("Sorry, I'm busy at the moment")
            return

        self.running_gen = True
        try:
            async with ctx.typing():
                try:
                    # Run the generation asynchronously so our Discord connection stays active
                    generate_text = functools.partial(
                        self._handle_cmd, Command.GENERATE, seed_text
                    )
                    output = await self.bot.loop.run_in_executor(None, generate_text)
                    await self.channel.send(output)
                except InvalidCommandError as e:
                    raise commands.BadArgument(str(e))
        finally:
            self.running_gen = False


class DeathsaurusImage(Deathsaurus):
    def __init__(
        self,
        bot: commands.Bot,
        guild: str,
        channel: str,
        hof_channel: str,
        dalle_model,
        dalle_params,
        vqgan,
        vqgan_params,
        dalle_processor,
    ):
        super().__init__(bot, guild, channel, hof_channel)
        self.dalle_model = dalle_model
        self.dalle_params = dalle_params
        self.vqgan = vqgan
        self.vqgan_params = vqgan_params
        self.dalle_processor = dalle_processor

        self.running_gen = False

    @property
    def name(self):
        return "Deathsaurus for Image Generation"

    def _handle_cmd(self, cmd: Command, text: str) -> str | list[io.BytesIO]:
        return handle_cmd_image(
            cmd,
            text,
            self.dalle_model,
            self.dalle_params,
            self.vqgan,
            self.vqgan_params,
            self.dalle_processor,
        )

    @commands.command(help="Generate images from the given seed text.")
    @commands.guild_only()
    async def gen(self, ctx, *, seed_text: str):
        if self.running_gen:
            await self.channel.send("Sorry, I'm busy at the moment")
            return

        self.running_gen = True
        try:
            async with ctx.typing():
                try:
                    # Run generation asynchronously so our Discord connection stays active
                    generate_images = functools.partial(
                        self._handle_cmd, Command.GENERATE, seed_text
                    )

                    output = await self.bot.loop.run_in_executor(None, generate_images)
                    if isinstance(output, str):
                        await self.channel.send(output)
                    elif isinstance(output, list):
                        for i, img in enumerate(output):
                            filename = f"{seed_text} ({i})"
                            await self.channel.send(
                                filename,
                                file=discord.File(img, filename=f"{filename}.png"),
                            )
                        await self.channel.send("Done generating images.")

                except InvalidCommandError as e:
                    raise commands.BadArgument(str(e))
        finally:
            self.running_gen = False


def get_text_bot(
    guild: str,
    channel: str,
    hof_channel: str,
    model: nn.Module,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    device: torch.device,
) -> commands.Bot:
    """
    Initialize and return a Discord bot that can listen to and respond to
    commands for text generation.

    Args:
      guild: Discord server to listen on.
      channel: Discord channel to post output in.
      hof_channel: Discord channel to post "hall of fame" output in.
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
    bot.add_cog(
        DeathsaurusText(bot, guild, channel, hof_channel, model, tokenizer, device)
    )
    return bot


def get_image_bot(
    guild: str,
    channel: str,
    hof_channel: str,
    dalle_model,
    dalle_params,
    vqgan,
    vqgan_params,
    dalle_processor,
) -> commands.Bot:
    bot = commands.Bot(
        command_prefix=commands.when_mentioned_or(COMMAND_PREFIX),
        description="Bot for interfacing with DALL-E Mini neural network models for whacky hijinks.",
    )
    bot.add_cog(
        DeathsaurusImage(
            bot,
            guild,
            channel,
            hof_channel,
            dalle_model,
            dalle_params,
            vqgan,
            vqgan_params,
            dalle_processor,
        )
    )
    return bot
