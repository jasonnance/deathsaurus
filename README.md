# Deathsaurus

Discord bot to interface with Transformer models.

## Prerequisites

Requires Docker, NVIDIA driver v440+ with a supported NVIDIA GPU, and the NVIDIA container runtime for Docker.  Could probably run on a CPU without all the NVIDIA stuff, but you'd have to take all the GPU-specific stuff out of the Docker invocations in the Makefile (and it would be _really_ slow).  You'll also need an [NVIDIA GPU Cloud](https://www.nvidia.com/en-us/gpu-cloud/) account to pull the base Docker image.

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
- `!xfm_gen <text>`: Generate text from the given seed
