.DEFAULT_GOAL := help
PY ?= .venv/bin/python
IMAGE ?= property-ai-assistant
CHROME ?= google-chrome

.PHONY: help install sitemap data index index-full run test smoke showcase lint format check docker-build docker-run

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install runtime + dev dependencies
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements-dev.txt

sitemap:  ## Re-scrape the Wasalt sitemap index (data/raw)
	$(PY) scraper/wasalt_sitemap.py

data:  ## Build cleaned documents from data/raw -> data/processed
	$(PY) -m pipeline.build_documents

index: data  ## Embed new/changed documents and write data/index
	$(PY) -m pipeline.build_index

index-full: data  ## Re-embed every document (a few minutes on CPU)
	$(PY) -m pipeline.build_index --full

run:  ## Run the app locally with auto-reload on http://localhost:7860
	$(PY) -m uvicorn app.main:app --reload --port 7860

test:  ## Run the test suite
	$(PY) -m pytest

smoke:  ## End-to-end checks against a running app: make smoke URL=https://...
	$(PY) scripts/smoke_test.py $(URL)

showcase:  ## Rebuild the 3-page showcase PDF from docs/showcase/showcase.html (needs Chrome)
	$(CHROME) --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer --allow-file-access-from-files \
		--print-to-pdf=docs/Property-AI-Assistant-Showcase.pdf "file://$(CURDIR)/docs/showcase/showcase.html"

lint:  ## Lint and check formatting
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Auto-format and fix lint issues
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

check: lint test  ## Everything CI would run

docker-build:  ## Build the production image
	docker build -t $(IMAGE) .

docker-run:  ## Run the image on http://localhost:7860 using .env
	docker run --rm -p 7860:7860 --env-file .env $(IMAGE)
