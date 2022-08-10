.PHONY: build run-local run-discord mkdirs ci

ROOT_DIR := $(shell pwd)
TRANSFORMER_CACHE_DIR := "$(ROOT_DIR)/data/transformer_cache"
DIST_DIR := "$(ROOT_DIR)/dist"

mkdirs:
	@mkdir -p $(TRANSFORMER_CACHE_DIR)
	@mkdir -p $(DIST_DIR)

build: mkdirs
	pip install -r requirements.txt
	pip install -e .

run-local-text:
	deathsaurus \
		--mode text \
		--run-local \
		--cache-dir $(TRANSFORMER_CACHE_DIR) \
		--model-name gpt2-large

run-local-image:
	deathsaurus \
		--mode image \
		--run-local

discord-vars:
ifndef DISCORD_BOT_TOKEN
	$(error "Set the DISCORD_BOT_TOKEN variable to the bot token for your bot application.")
endif
ifndef DISCORD_BOT_GUILD
	$(error "Set the DISCORD_BOT_GUILD variable to the server name you want your bot to listen to.")
endif
ifndef DISCORD_BOT_CHANNEL
	$(error "Set the DISCORD_BOT_CHANNEL variable to the channel name you want your bot to post in.")
endif

run-discord-text: discord-vars
	deathsaurus \
		--mode text \
		--run-discord \
		--cache-dir $(TRANSFORMER_CACHE_DIR) \
		--model-name gpt2-large

run-discord-image: discord-vars
	deathsaurus \
		--mode image \
		--run-discord

ci: build
	./run_ci.sh
