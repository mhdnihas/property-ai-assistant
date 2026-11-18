"""Build a Wasalt listing index from its public sitemaps.

wasalt.sa's HTML pages are behind a Cloudflare challenge for scripts, but the
sitemaps it advertises in robots.txt live on cdn.wasalt.sa and are plain XML.
Each listing URL is descriptive, e.g.

    /en/property/sale/villa-333-sqm-facing-east-on-20m-width-street-5841671

so we can recover purpose (sale/rent), property type, area, bedrooms, facing,
street width and the listing ID — no price or city (those come from the
browser-extracted category pages, see wasalt_extract.js).

Usage:
    python scraper/wasalt_sitemap.py            # writes data/raw/wasalt_sitemap_index.jsonl
"""

from __future__ import annotations

import contextlib
import gzip
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

SITEMAPS = [
    "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
]
UA = "Mozilla/5.0 (compatible; property-ai-assistant/1.0; research project)"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "wasalt_sitemap_index.jsonl"

LOC_RE = re.compile(r"<loc>([^<]+)</loc>\s*(?:<lastmod>([^<]+)</lastmod>)?")
SLUG_RE = re.compile(r"/en/property/(?P<purpose>sale|rent)/(?P<slug>.+)-(?P<id>\d+)$")
TYPE_RE = re.compile(r"^(?P<type>[a-z]+(?:-[a-z]+)*?)-(?P<area>\d+)-sqm")
BEDS_RE = re.compile(r"with-(\d+)-bedrooms?")
FACING_RE = re.compile(r"facing-([a-z]+(?:-[a-z]+)?)(?:-on|$)")
STREET_RE = re.compile(r"on-(\d+)m-width-street")


def fetch(url: str) -> str:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL: {url}")
    req = Request(url, headers={"User-Agent": UA})  # noqa: S310 (scheme checked above)
    with urlopen(req, timeout=60) as r:  # noqa: S310
        raw = r.read()
    with contextlib.suppress(OSError):  # CDN may already serve it decompressed
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def parse_url(url: str, lastmod: str | None) -> dict | None:
    m = SLUG_RE.search(url)
    if not m:
        return None
    slug = m["slug"]
    rec = {"id": int(m["id"]), "url": url, "purpose": m["purpose"], "lastmod": lastmod}
    if t := TYPE_RE.match(slug):
        rec["type"] = t["type"].replace("-", " ")
        # Note: the slug drops decimal points (248.35 sqm -> "24835"), so area is
        # only reliable for whole numbers. Kept raw and flagged.
        rec["area_sqm_raw"] = int(t["area"])
    if b := BEDS_RE.search(slug):
        rec["bedrooms"] = int(b[1])
    if f := FACING_RE.search(slug):
        rec["facing"] = f[1].replace("-", " ")
    if s := STREET_RE.search(slug):
        rec["street_width_m"] = int(s[1])
    return rec


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for sm in SITEMAPS:
            print(f"fetching {sm}", file=sys.stderr)
            xml = fetch(sm)
            for url, lastmod in LOC_RE.findall(xml):
                if rec := parse_url(url.strip(), lastmod or None):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            time.sleep(1)
    print(f"wrote {n} listings -> {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
