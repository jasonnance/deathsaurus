.PHONY: build run-local run-discord mkdirs ci

IMAGE_TAG := "deathsaurus"
ROOT_DIR := $(shell pwd)
TRANSFORMER_CACHE_DIR := "$(ROOT_DIR)/data/transformer_cache"

mkdirs:
	@mkdir -p $(TRANSFORMER_CACHE_DIR)

build:
	docker build -t $(IMAGE_TAG) .

run-local: build mkdirs
	docker run --rm \
		--ipc host \
		--gpus all \
		-it \
		-v $(TRANSFORMER_CACHE_DIR):/cache \
		$(IMAGE_TAG) \
		python deathsaurus.py \
			--cache-dir /cache \
			--run-local

run-discord: build mkdirs
ifndef DISCORD_BOT_TOKEN
	$(error "Set the DISCORD_BOT_TOKEN variable to the bot token for your bot application.")
endif
ifndef DISCORD_BOT_GUILD
	$(error "Set the DISCORD_BOT_GUILD variable to the server name you want your bot to listen to.")
endif
ifndef DISCORD_BOT_CHANNEL
	$(error "Set the DISCORD_BOT_CHANNEL variable to the channel name you want your bot to post in.")
endif

	docker run --rm \
		--ipc host \
		--gpus all \
		-it \
		-v $(TRANSFORMER_CACHE_DIR):/cache \
		-e DISCORD_BOT_TOKEN=$(DISCORD_BOT_TOKEN) \
		-e DISCORD_BOT_GUILD=$(DISCORD_BOT_GUILD) \
		-e DISCORD_BOT_CHANNEL=$(DISCORD_BOT_CHANNEL) \
		$(IMAGE_TAG) \
		python deathsaurus.py \
			--cache-dir /cache \
			--run-discord

ci: build
	docker run --rm \
		-it \
		$(IMAGE_TAG) \
		./run_ci.sh
