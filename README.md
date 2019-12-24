# Deathsaurus

Discord bot to interface with Transformer models.

## Prerequisites

Requires Docker.  It's strongly recommended to have NVIDIA driver v440+ with a supported NVIDIA GPU and the NVIDIA container runtime for Docker.  You'll also need an [NVIDIA GPU Cloud](https://www.nvidia.com/en-us/gpu-cloud/) account to pull the base Docker image.  The Makefile (hackily) attempts to use a GPU if one is available.

## Usage

Make sure you're logged in before pulling the base image:

    docker login nvcr.io

Build the Docker image:

    make build
    
Run the Docker image to get a local REPL for bot commands:

    make run-local
    
Export some needed environment variables and run the bot connected to Discord:

    export DISCORD_BOT_TOKEN="<your token>"
    export DISCORD_BOT_GUILD="<your server name>"
    export DISCORD_BOT_CHANNEL="<channel where bot should post>"
    make run-discord
    
## Commands

- `!xfm_help`: Display usage
- `!xfm_ping`: Ping the bot to check if it's online
- `!xfm_gen <text>`: Generate text from the given seed

## Parameter Guidelines

### Text Generation

For the GPT2-Large model:

- `temperature = 0.1`: Boring, very realistic text
- `temperature = 0.7`: Mostly realistic text
- `temperature = 0.99`: Coherent yet very whacky text
- `temperature > 5`: Incoherent mass of words
