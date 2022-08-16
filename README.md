# Deathsaurus

Discord bot to interface with Transformer models.

## Prerequisites

You need an NVIDIA GPU to run the models. Requires Python 3.10+ and the appropriate CUDA libraries installed on your system (including cuDNN). Tested in WSL2 (Ubuntu) on Windows 10.

Jax can't be installed directly by pip in all cases -- see [instructions here](https://github.com/google/jax#installation).

You may need to set the following environment variable:

```
# You may get an error about compilation parallelism if this isn't set
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

# Make sure your CUDA libraries are on LD_LIBRARY_PATH --
# this is the default install location for WSL2
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
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
    export DISCORD_BOT_HOF_CHANNEL="<channel where bot should make hall-of-fame posts>"

    # Text generation
    make run-discord-text

    # Image generation
    make run-discord-image

## Commands

- `!deathsaurus_help`: Display usage
- `!deathsaurus_ping`: Ping the bot to check if it's online
- `!deathsaurus_gen <text>`: Generate text from the given seed
- `!deathsaurus_hof`: Print instructions on using the Hall of Fame

You can also @ the bot to run commands:

- `@<bot_name> help`
- `@<bot_name> ping`
- `@<bot_name> gen`
- `@<bot_name> hof`

## Parameter Guidelines

### Text Generation

For the GPT2-Large model:

- `temperature = 0.1`: Boring, very realistic text
- `temperature = 0.7`: Mostly realistic text
- `temperature = 0.99`: Coherent yet very whacky text
- `temperature > 5`: Incoherent mass of words
