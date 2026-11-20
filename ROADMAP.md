# Roadmap: DarGlobal Junior AI Engineer assignment

**Deadline:** Friday 20 Nov 2026. Reply to careers@darglobal.co.uk.

**Assignment:** scrape DarGlobal and Wasalt → build an AI chatbot on that data → use a free
OpenRouter model → containerise with Docker → deploy → send a working URL.

| When | Work | Status |
|---|---|---|
| Wed 18 | Scrape DarGlobal (65 pages) and Wasalt (32 category pages + 60k-URL sitemap) | ✅ Done |
| Wed 18 | Data pipeline: dedupe, chunk, normalise → 1,484 documents (incl. per-country project directory) | ✅ Done |
| Wed 18 | Hybrid search index (BM25 + bge-small embeddings + listing filters) | ✅ Done |
| Wed 18 | FastAPI backend: SSE streaming, OpenRouter with model fallback, rate limiting | ✅ Done |
| Wed 18 | Chat web UI: streaming, citations, source cards, mobile, dark mode | ✅ Done |
| Wed 18 / Thu 19 | Tests (62 unit + 7-check smoke test), lint, Docker image (non-root, health check) | ✅ Done |
| Thu 19 | Live test with a real OpenRouter key → fixed truncation, rate-limit cool-down, Oman completeness, atomic index | ✅ Done |
| Thu 19 / Fri 20 | Deploy: Hugging Face Docker is paid and Koyeb closed to new accounts, so deployed on **Render** → https://property-ai-assistant.onrender.com | ✅ Done |
| Fri 20 | Test the live URL end to end: smoke test 7/7 + browser UI on desktop | ✅ Done (phone not yet checked) |
| Fri 20 | Live URL in README and docs, code on GitHub | ✅ Done (push the latest commit) |
| Fri 20 AM | Send the email: location, current/expected package, notice period, live URL, repo link | ⬜ |

## Stretch (only if time allows)

- Arabic questions (multilingual embeddings + Arabic sitemaps)
- A small evaluation set of questions with expected sources
- Scheduled re-scrape and re-index

## Things only you can do

- Keep the OpenRouter key in `.env` (local) and in Render's environment variables (production)
- Optional: keep the app warm during the review with an uptime monitor on `/api/health`
- Fill in your personal details in the reply email
