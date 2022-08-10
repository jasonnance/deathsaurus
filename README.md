# Deathsaurus

Discord bot to interface with Transformer models.

## Prerequisites

You need an NVIDIA GPU to run the models. Requires Python 3.10+ and the appropriate CUDA libraries installed on your system (including cuDNN). Tested in WSLv2 (Ubuntu) on Windows 10.

Jax can't be installed directly by pip in all cases -- see [instructions here](https://github.com/google/jax#installation).

You may need to set the following environment variable:

```
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
```

## Usage

Build the requirements:

    make build

Run a REPL for bot commands:

    # Text generation
    make run-local-text

    # Image generation
    make run-local-image

Export some needed environment variables and run the bot connected to Discord:

    export DISCORD_BOT_TOKEN="<your token>"
    export DISCORD_BOT_GUILD="<your server name>"
    export DISCORD_BOT_CHANNEL="<channel where bot should post>"

    # Text generation
    make run-discord-text

    # Image generation
    make run-discord-image

## Commands

- `!xfm_help`: Display usage
- `!xfm_ping`: Ping the bot to check if it's online
- `!xfm_gen <text>`: Generate text from the given seed

You can also @ the bot to run commands:

- `@<bot_name> help`
- `@<bot_name> ping`
- `@<bot_name> gen`

## Parameter Guidelines

### Text Generation

For the GPT2-Large model:

- `temperature = 0.1`: Boring, very realistic text
- `temperature = 0.7`: Mostly realistic text
- `temperature = 0.99`: Coherent yet very whacky text
- `temperature > 5`: Incoherent mass of words
