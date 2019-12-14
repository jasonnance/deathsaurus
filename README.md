# Deathsaurus

Discord bot to interface with Transformer models.

## Prerequisites

Requires Docker, NVIDIA driver v440+ with a supported NVIDIA GPU, and the NVIDIA container runtime for Docker.  Could probably run on a CPU without all the NVIDIA stuff, but you'd have to take all the GPU-specific stuff out of the Docker invocations in the Makefile (and it would be _really_ slow).

## Usage

Build the Docker image:

    make build
    
Run the Docker image:

    make run
