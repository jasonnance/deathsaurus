.PHONY: build run-local run-discord mkdirs ci

ROOT_DIR := $(shell pwd)
TRANSFORMER_CACHE_DIR := "$(ROOT_DIR)/data/transformer_cache"
DIST_DIR := "$(ROOT_DIR)/dist"

mkdirs:
	@mkdir -p $(TRANSFORMER_CACHE_DIR)
	@mkdir -p $(DIST_DIR)

build:
	pip install -r requirements.txt
	pip install -e .

# TODO put back
#run-local: build mkdirs
run-local:
	deathsaurus \
		--run-local \
		--cache-dir $(TRANSFORMER_CACHE_DIR) \
		--model-name gpt2-large

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
		$(GPU_ARGS) \
		-it \
		-v $(TRANSFORMER_CACHE_DIR):/cache \
		-e DISCORD_BOT_TOKEN=$(DISCORD_BOT_TOKEN) \
		-e DISCORD_BOT_GUILD=$(DISCORD_BOT_GUILD) \
		-e DISCORD_BOT_CHANNEL=$(DISCORD_BOT_CHANNEL) \
		$(IMAGE_TAG) \
		deathsaurus \
			--cache-dir /cache \
			--run-discord \
			--model-name gpt2-large

ci: build
	./run_ci.sh
