.PHONY: build run mkdirs ci

IMAGE_TAG := "deathsaurus"
ROOT_DIR := $(shell pwd)
TRANSFORMER_CACHE_DIR := "$(ROOT_DIR)/data/transformer_cache"

mkdirs:
	@mkdir -p $(TRANSFORMER_CACHE_DIR)

build:
	docker build -t $(IMAGE_TAG) .

run: build mkdirs
	docker run --rm \
		--ipc host \
		--gpus all \
		-it \
		-v $(TRANSFORMER_CACHE_DIR):/cache \
		$(IMAGE_TAG) \
		python deathsaurus.py \
			--cache-dir /cache

ci: build
	docker run --rm \
		-it \
		$(IMAGE_TAG) \
		./run_ci.sh
