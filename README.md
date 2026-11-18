---
title: Property AI Assistant
emoji: 🏙️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: RAG chatbot over DarGlobal projects and Wasalt listings
---

# Property AI Assistant

A retrieval-augmented chatbot that answers questions about **DarGlobal** luxury projects
(darglobal.co.uk) and **Saudi property listings on Wasalt** (wasalt.sa). The data was scraped
from both public websites, and answers come from a **free OpenRouter model**. Every answer
cites its sources.

**Live demo:** _<your Hugging Face Space URL>_

![architecture](https://img.shields.io/badge/stack-FastAPI%20·%20BM25%20%2B%20bge--small%20·%20OpenRouter-9a7642)

## What it can answer

| Ask… | Answered from |
|---|---|
| "Tell me about Trump Tower Jeddah" / "When will W Residences be delivered?" | DarGlobal project pages |
| "Which DarGlobal projects are in Oman?" | DarGlobal project catalogue |
| "How can I get Saudi residency by investing?" | DarGlobal *Why Invest* + FAQ |
| "Villas for sale in Riyadh under 2 million" | Wasalt listings (structured filters) |
| "3 bedroom apartments for rent in Jeddah" → "what about Dammam?" | Wasalt listings + follow-up memory |
| "What's the average villa price in Riyadh?" | Wasalt FAQs and market guides |
| "How many listings does Wasalt have?" | Wasalt sitemap statistics |

## Architecture

```
 OFFLINE (scrape → clean → embed; outputs committed)        RUNTIME (one Docker container)
 ┌──────────────────────────────┐                          ┌───────────────────────────────────────┐
 │ scraper/darglobal_extract.js │  data/raw/*.json         │ FastAPI  (app/main.py)                │
 │ scraper/wasalt_extract.js    │ ───────────────┐         │   GET  /            chat UI           │
 │ scraper/wasalt_sitemap.py    │                │         │   POST /api/chat    SSE stream        │
 └──────────────────────────────┘                ▼         │   GET  /api/health  status            │
 ┌──────────────────────────────┐  data/processed/         │                                       │
 │ pipeline/build_documents.py  │  documents.jsonl         │ ChatService (app/rag/service.py)      │
 │  dedupe · chunk · normalise  │ ───────────────┐         │  1. parse filters  (query_parser.py)  │
 └──────────────────────────────┘                ▼         │  2. hybrid search  (retriever.py)     │
 ┌──────────────────────────────┐  data/index/   ────────► │     BM25 + bge-small, RRF fusion,     │
 │ pipeline/build_index.py      │  embeddings.npy          │     hard metadata filters             │
 │  fastembed bge-small (CPU)   │  manifest.json           │  3. grounded prompt (prompt.py)       │
 └──────────────────────────────┘                          │  4. OpenRouter stream with model      │
                                                           │     fallback (llm/openrouter.py)      │
                                                           └───────────────────────────────────────┘
```

### Key design decisions

- **Scrape offline, serve from a committed index.** The live demo never depends on the target
  sites being up, and the container starts in seconds. Embedding 1,477 documents takes about
  5 minutes on CPU, so it isn't repeated in every Docker build.
- **Hybrid retrieval.** BM25 handles exact names ("Pagani", "W Residences") and numbers. Dense
  embeddings (`BAAI/bge-small-en-v1.5` through fastembed/ONNX: free, no torch, no API) handle
  meaning. Reciprocal Rank Fusion combines the two rankings without needing score calibration.
- **Structured filters before the LLM.** A deterministic parser pulls out city, sale or rent,
  property type, bedrooms and price range. Listings that don't match are excluded **before**
  ranking, so "under 2 million" is a hard rule rather than a hint to the model. The prompt
  also tells the model how many listings matched, or that none did.
- **Grounded generation.** The system prompt restricts answers to the retrieved context,
  requires inline `[n]` citations, treats context as data (a defence against prompt
  injection), and tells the model to say so when it doesn't know.
- **Resilient LLM client.** Free models are often rate-limited. The client streams over SSE
  and falls back to the next model on 429/5xx/timeouts, but only *before* the first token
  reaches the user, so answers are never stitched together from two models. A rejected
  API key fails fast.
- **Production hygiene.** Settings come from environment variables, the key is held as a
  secret type and never logged, requests carry IDs, rate limiting is per IP, input sizes are
  limited, security headers are set, the Docker image is a multi-stage build that runs as a
  non-root user with a health check, and the UI renders model output with an
  HTML-escaping markdown renderer.

## Data collection

| Source | Method | What was collected |
|---|---|---|
| darglobal.co.uk | Browser extractor (`scraper/darglobal_extract.js`) | 65 pages → 49 unique: every project, the catalogue by country, FAQ, About, Why Invest |
| wasalt.sa pages | Browser extractor (`scraper/wasalt_extract.js`) | 32 category pages (6 cities × villas/apartments/land, sale/rent): 900 unique listings with price, district, bedrooms and area, plus 111 FAQs and market guides |
| wasalt.sa sitemap | `scraper/wasalt_sitemap.py` | 60,000 listing URLs → inventory statistics |

**Responsible scraping:** both sites block non-browser HTTP clients (Imperva on DarGlobal,
Cloudflare on Wasalt). These protections were **not** bypassed. Pages were read from a normal
browser session, at one page every 1.5–2 seconds, and only from paths `robots.txt` allows.
Wasalt's sitemaps are public and listed in its `robots.txt`. Each extractor reads the page's
own Next.js `__NEXT_DATA__` JSON, which is more reliable than parsing HTML.

Known data caveats: the Wasalt data is a sample (the first page of each category, about
900 of roughly 60k listings) captured on 18 Nov 2026. Some listing prices are clearly
placeholders set by the owner. Sitemap areas lose their decimal point, so they are only
used for counts.

## Run locally

```bash
make install            # .venv + dependencies
cp .env.example .env    # then set OPENROUTER_API_KEY (free key: https://openrouter.ai/keys)
make run                # http://localhost:7860
```

With Docker:

```bash
make docker-build
make docker-run         # uses .env
```

Rebuild the data:

```bash
make sitemap   # re-fetch the Wasalt sitemap index
make index     # data/raw -> documents -> embeddings (a few minutes on CPU)
```

The browser extractors are run by pasting them into the DevTools console on the target site.
They download a JSON file, which then goes in `data/raw/`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Required** for answers |
| `OPENROUTER_MODELS` | `nvidia/nemotron-3-ultra-550b-a55b:free,nvidia/nemotron-3.5-super-vl:free,google/gemma-4-31b-it:free` | Tried in order |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.2` / `1500` | Generation settings |
| `LLM_DISABLE_REASONING` | `true` | Turn off hidden "thinking" tokens; they consume the answer budget and add latency |
| `ENABLE_EMBEDDINGS` | `true` | `false` = BM25-only (lighter) |
| `TOP_K` | `8` | Documents per answer |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per client IP |
| `PUBLIC_URL` | HF Spaces | Sent to OpenRouter as `HTTP-Referer` |

## Deploy to Hugging Face Spaces

1. Create a Space at huggingface.co/new-space → SDK **Docker** → Blank → Public.
2. Add the secret: Space **Settings → Variables and secrets → New secret** →
   `OPENROUTER_API_KEY`.
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
   Use a Hugging Face access token with *write* scope as the password.
4. The Space builds the Dockerfile (about 3–4 minutes) and serves the app at
   `https://<user>-<space-name>.hf.space`.

## Quality checks

```bash
make check     # ruff lint + format check + pytest
```

There are 44 tests covering the query parser, the data pipeline (deduplication, chunking,
normalisation), retrieval (filters, fusion), the OpenRouter client (streaming, fallback, error
handling, using a mocked transport) and the API (SSE contract, history, validation, rate
limiting, missing-key path, shutdown).

## Project layout

```
app/
  config.py            settings (pydantic-settings)
  main.py              FastAPI app factory, SSE endpoint, middleware, rate limiter
  llm/openrouter.py    streaming client with model fallback
  rag/                 models, vocab, query_parser, retriever, embeddings, prompt, service
  static/              chat UI (vanilla JS, safe markdown renderer)
pipeline/              build_documents.py, build_index.py
scraper/               browser extractors + sitemap scraper
data/raw/              scraped data (committed)
data/index/            search index (committed; rebuilt with `make index`)
tests/                 pytest suite
```

## Possible next steps

- Scheduled re-scrape and re-index (for example a weekly GitHub Action) to keep listings fresh
- Arabic questions: Wasalt publishes Arabic sitemaps, and a multilingual embedding model
  (`multilingual-e5-small`) would make them work
- An evaluation set of questions with expected sources, tracked in CI
- Redis-backed rate limiting and response caching when running more than one replica
