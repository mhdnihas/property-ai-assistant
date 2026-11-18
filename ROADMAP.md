# Roadmap: DarGlobal Junior AI Engineer assignment

**Deadline:** Friday 20 Nov 2026. Reply to careers@darglobal.co.uk.

**Assignment:** scrape DarGlobal and Wasalt → build an AI chatbot on that data → use a free
OpenRouter model → containerise with Docker → deploy → send a working URL.

| When | Work | Status |
|---|---|---|
| Wed 18 | Scrape DarGlobal (65 pages) and Wasalt (32 category pages + 60k-URL sitemap) | ✅ Done |
| Wed 18 | Data pipeline: dedupe, chunk, normalise → 1,477 documents | ✅ Done |
| Wed 18 | Hybrid search index (BM25 + bge-small embeddings + listing filters) | ✅ Done |
| Wed 18 | FastAPI backend: SSE streaming, OpenRouter with model fallback, rate limiting | ✅ Done |
| Wed 18 | Chat web UI: streaming, citations, source cards, mobile, dark mode | ✅ Done |
| Wed 18 | Tests (44), lint, Docker image (non-root, health check) | ✅ Done |
| Wed 18 / Thu 19 | Live test with a real OpenRouter key | ⬜ Needs key in `.env` |
| Thu 19 | Deploy to Hugging Face Spaces, add `OPENROUTER_API_KEY` as a Space secret | ⬜ Needs HF account |
| Thu 19 | Test the live URL end to end (desktop + phone) | ⬜ |
| Fri 20 AM | Put the live URL in the README, push the code to GitHub | ⬜ |
| Fri 20 AM | Send the email: location, current/expected package, notice period, live URL, repo link | ⬜ |

## Stretch (only if time allows)

- Arabic questions (multilingual embeddings + Arabic sitemaps)
- A small evaluation set of questions with expected sources
- Scheduled re-scrape and re-index

## Things only you can do

- Create the Hugging Face account and Space, and a write token for `git push`
- Add the OpenRouter key to `.env` (local) and as a Space secret (production)
- Fill in your personal details in the reply email
