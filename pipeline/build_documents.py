"""Turn raw scraped data (data/raw) into clean, chunked documents (data/processed).

    python -m pipeline.build_documents

Steps
  * DarGlobal: drop error/empty pages, de-duplicate pages whose text is identical
    (client-rendered routes return the same payload), split long pages into
    overlapping chunks on the CMS field separator.
  * Wasalt pages: one document per listing (with structured metadata used for
    filtering), one per FAQ entry, chunked market overviews.
  * Wasalt sitemap: aggregated into a single inventory-statistics document; the
    60k individual URLs carry no price/city, so indexing them would add noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR
from app.rag.models import Document
from app.rag.vocab import normalize_city, normalize_type

log = logging.getLogger("pipeline.documents")

RAW_DIR = ROOT_DIR / "data" / "raw"
OUT_PATH = ROOT_DIR / "data" / "processed" / "documents.jsonl"

MAX_CHUNK_CHARS = 1400
MIN_PAGE_CHARS = 150

# Boilerplate fragments repeated on most DarGlobal pages; they add noise to every chunk.
DG_BOILERPLATE = re.compile(
    r"^(view all images|download brochure|book unit|book now|get in touch|register your interest|"
    r"find your next investment|enquire now|read more|learn more|explore|submit|gallery|close)$",
    re.I,
)


# --------------------------------------------------------------------------- helpers
def _id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]  # noqa: S324 (not security)


def chunk_segments(segments: list[str], max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Greedily pack text segments into chunks <= max_chars with one segment of overlap.

    Segments longer than max_chars are split on sentence boundaries first.
    """
    pieces: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            pieces.append(seg)
            continue
        buf = ""
        for sentence in re.split(r"(?<=[.!?])\s+", seg):
            if buf and len(buf) + len(sentence) + 1 > max_chars:
                pieces.append(buf)
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
        if buf:
            pieces.append(buf[: max_chars * 2])

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) + 3 > max_chars:
            chunks.append(" | ".join(current))
            last = current[-1]
            current, size = ([last], len(last)) if len(last) < max_chars // 3 else ([], 0)
        current.append(piece)
        size += len(piece) + 3
    if current:
        chunks.append(" | ".join(current))
    return chunks


def _fmt_money(value: float | int | None, currency: str = "SAR") -> str | None:
    if value is None:
        return None
    return f"{value:,.0f} {currency}"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def _area(value: Any) -> float | None:
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- DarGlobal
def _dg_kind(path: str) -> str:
    if path.startswith("/projects"):
        return "catalogue"
    info_pages = {
        "/",
        "/about",
        "/faq",
        "/why-invest",
        "/investor",
        "/hospitality",
        "/development-management",
        "/partners",
        "/one-of-one",
        "/luxury-golf-communities",
        "/live-all-in",
        "/commercial",
    }
    return "page" if path in info_pages else "project"


def _dg_title(page: dict[str, Any]) -> str:
    title = (page.get("title") or page["path"]).strip()
    # "Trump Tower Jeddah | Seafront Residences by DarGlobal" -> keep both halves, drop brand suffix
    return re.sub(r"\s*[|-]\s*(Dar ?Global|Darglobal)\s*$", "", title, flags=re.I) or title


def darglobal_documents(raw: dict[str, Any]) -> Iterator[Document]:
    seen_hashes: set[str] = set()
    for page in raw["pages"]:
        text = (page.get("text") or "").strip()
        if page.get("status") != 200 or page.get("error") or len(text) < MIN_PAGE_CHARS:
            log.info(
                "skip darglobal %s (status=%s, %d chars)",
                page["path"],
                page.get("status"),
                len(text),
            )
            continue
        digest = hashlib.sha1(text.encode()).hexdigest()  # noqa: S324
        if digest in seen_hashes:
            log.info("skip darglobal %s (duplicate content)", page["path"])
            continue
        seen_hashes.add(digest)

        segments = [
            s.strip()
            for s in text.split(" | ")
            if s.strip() and not DG_BOILERPLATE.match(s.strip())
        ]
        desc = (page.get("desc") or "").strip()
        if desc and desc not in segments:
            segments.insert(0, desc)
        title = _dg_title(page)
        kind = _dg_kind(page["path"])
        chunks = chunk_segments(segments)
        for i, chunk in enumerate(chunks):
            yield Document(
                id=_id("dg", page["path"], i),
                source="darglobal",
                kind=kind,  # type: ignore[arg-type]
                title=title if len(chunks) == 1 else f"{title} (part {i + 1}/{len(chunks)})",
                url=page["url"],
                text=chunk,
                meta={"path": page["path"], "chunk": i},
            )


# Country headings used on darglobal.co.uk/projects, and each country's listing page.
DG_COUNTRIES: dict[str, str] = {
    "United Arab Emirates": "properties-in-uae",
    "Spain": "properties-in-spain",
    "Qatar": "properties-in-qatar",
    "Oman": "properties-in-oman",
    "United Kingdom": "properties-in-uk",
    "Saudi Arabia": "properties-in-saudi-arabia",
}


def project_directory_documents(raw: dict[str, Any]) -> Iterator[Document]:
    """Build 'DarGlobal projects in <country>' documents from the /projects catalogue.

    The catalogue page starts with every project grouped under a country heading
    (name, location, unit types, delivery date). Generic chunking scatters that
    list across chunks, so country questions ("which projects are in Oman?") only
    saw part of it. One document per country keeps each list whole.
    """
    page = next((p for p in raw["pages"] if p.get("path") == "/projects" and p.get("text")), None)
    if page is None:
        return
    segments = [s.strip() for s in page["text"].split(" | ") if s.strip()]
    end = next((i for i, s in enumerate(segments) if s.startswith("More about")), len(segments))

    groups: dict[str, list[str]] = {}
    current: str | None = None
    for seg in segments[:end]:
        # A country name inside its own section is a location ("AIDA | Oman"), not a heading.
        if seg in DG_COUNTRIES and seg != current:
            current = seg
            groups.setdefault(seg, [])
        elif current is not None:
            groups[current].append(seg)

    for country, items in groups.items():
        if not items:
            continue
        yield Document(
            id=_id("dgdir", country),
            source="darglobal",
            kind="catalogue",
            title=f"DarGlobal projects in {country}",
            url=f"https://darglobal.co.uk/projects/{DG_COUNTRIES[country]}",
            text=(
                f"DarGlobal projects in {country}, from the DarGlobal project catalogue "
                "(project name, location, unit types and delivery date where listed): "
                + " | ".join(items)
            ),
            meta={"country": country},
        )

    if groups:
        summary = "; ".join(f"{c}: " + ", ".join(items) for c, items in groups.items() if items)
        yield Document(
            id=_id("dgdir", "all"),
            source="darglobal",
            kind="catalogue",
            title="DarGlobal project directory (all countries)",
            url=page["url"],
            text=(
                f"DarGlobal develops projects in {len(groups)} countries: "
                f"{', '.join(groups)}. Full directory — {summary}"
            ),
            meta={"countries": list(groups)},
        )


# --------------------------------------------------------------------------- Wasalt
def _rent_text(listing: dict[str, Any]) -> str | None:
    freq = listing.get("rent_frequency") or {}
    parts = [
        f"{v['amount']:,.0f} {v.get('enCurrencyType', 'SAR')} {k}"
        for k, v in freq.items()
        if isinstance(v, dict) and v.get("amount")
    ]
    return ", ".join(parts) or None


def listing_document(listing: dict[str, Any]) -> Document | None:
    if not listing.get("id") or not listing.get("url"):
        return None  # ad slots / placeholders
    attrs: dict[str, Any] = listing.get("attributes") or {}
    city = normalize_city(listing.get("city"))
    ptype = normalize_type(listing.get("type"))
    purpose = listing.get("purpose")
    bedrooms = _int_or_none(attrs.get("noOfBedrooms"))
    bathrooms = _int_or_none(attrs.get("noOfBathrooms"))
    land_area = _area(attrs.get("carpetArea"))
    built_area = _area(attrs.get("builtUpArea"))
    price = listing.get("price")
    currency = listing.get("currency") or "SAR"

    lines = [
        f"{listing.get('title') or ptype} — {ptype or 'Property'} for {purpose} in "
        f"{listing.get('address') or city}."
    ]
    if purpose == "rent":
        rent = _rent_text(listing)
        lines.append(f"Rent: {rent or _fmt_money(price, currency)}.")
    elif price:
        lines.append(f"Price: {_fmt_money(price, currency)}.")
    if bedrooms is not None:
        lines.append(f"Bedrooms: {bedrooms}.")
    if bathrooms is not None:
        lines.append(f"Bathrooms: {bathrooms}.")
    if land_area:
        lines.append(f"Land area: {land_area:g} sqm.")
    if built_area:
        lines.append(f"Built-up area: {built_area:g} sqm.")
    if land_area and price and purpose == "sale":
        lines.append(f"Price per sqm: {price / land_area:,.0f} {currency}.")
    if age := attrs.get("completionYear"):
        lines.append(f"Age: {'new build' if str(age).lower() == 'new' else age}.")
    if listing.get("furnishing"):
        lines.append(f"Furnishing: {listing['furnishing']}.")
    if listing.get("amenities"):
        lines.append("Amenities: " + ", ".join(listing["amenities"]) + ".")
    if listing.get("rega_licensed"):
        lines.append("REGA-licensed advertisement.")
    if listing.get("published_at"):
        lines.append(f"Listed on {listing['published_at'][:10]}.")

    return Document(
        id=_id("wl", listing["id"]),
        source="wasalt",
        kind="listing",
        title=listing.get("title") or f"{ptype} for {purpose}",
        url=listing["url"],
        text=" ".join(lines),
        meta={
            "listing_id": listing["id"],
            "city": city,
            "district": listing.get("district"),
            "purpose": purpose,
            "property_type": ptype,
            "bedrooms": bedrooms,
            "price": float(price) if price else None,
            "land_area_sqm": land_area,
            "built_area_sqm": built_area,
            "currency": currency,
        },
    )


def wasalt_documents(raw: dict[str, Any]) -> Iterator[Document]:
    seen_listings: set[int] = set()
    seen_faq: set[str] = set()
    for page in raw["pages"]:
        if page.get("error"):
            continue
        title = page.get("title") or page["path"]
        if page.get("total_count"):
            meta_line = f"{title}: {page['total_count']:,} listings on Wasalt at time of scraping."
        else:
            meta_line = title

        overview = (page.get("overview") or "").strip()
        if overview:
            chunks = chunk_segments([meta_line, *re.split(r"(?<=[.!?])\s+(?=[A-Z])", overview)])
            for i, chunk in enumerate(chunks):
                yield Document(
                    id=_id("wo", page["path"], i),
                    source="wasalt",
                    kind="market_guide",
                    title=f"{title} — market guide"
                    + (f" (part {i + 1})" if len(chunks) > 1 else ""),
                    url=page["url"],
                    text=chunk,
                    meta={"path": page["path"], "total_count": page.get("total_count")},
                )

        for qa in page.get("faq") or []:
            key = qa["q"].strip().lower()
            if not qa.get("a") or key in seen_faq:
                continue
            seen_faq.add(key)
            yield Document(
                id=_id("wf", key),
                source="wasalt",
                kind="faq",
                title=qa["q"].strip(),
                url=page["url"],
                text=f"Q: {qa['q'].strip()} A: {qa['a'].strip()}",
                meta={"path": page["path"]},
            )

        for listing in page.get("listings") or []:
            doc = listing_document(listing)
            if doc is None or doc.meta["listing_id"] in seen_listings:
                continue
            seen_listings.add(doc.meta["listing_id"])
            yield doc


def sitemap_stats_document(records: Iterable[dict[str, Any]]) -> Document:
    by_purpose_type: Counter[tuple[str, str]] = Counter()
    bedrooms: Counter[int] = Counter()
    total = 0
    for r in records:
        total += 1
        by_purpose_type[(r["purpose"], r.get("type") or "other")] += 1
        if "bedrooms" in r:
            bedrooms[r["bedrooms"]] += 1
    lines = [f"Wasalt public sitemap snapshot: {total:,} active property listings in Saudi Arabia."]
    for purpose in ("sale", "rent"):
        items = sorted(
            ((t, n) for (p, t), n in by_purpose_type.items() if p == purpose), key=lambda x: -x[1]
        )
        n_purpose = sum(n for _, n in items)
        lines.append(
            f"For {purpose}: {n_purpose:,} listings — "
            + ", ".join(f"{t}: {n:,}" for t, n in items)
            + "."
        )
    common = ", ".join(f"{b} bedrooms: {n:,}" for b, n in sorted(bedrooms.items()) if n >= 100)
    lines.append(f"Listings by bedroom count: {common}.")
    return Document(
        id=_id("ws", "stats"),
        source="wasalt",
        kind="market_stats",
        title="Wasalt inventory statistics (all listings)",
        url="https://wasalt.sa/en",
        text=" ".join(lines),
        meta={"total": total},
    )


# --------------------------------------------------------------------------- main
def build(raw_dir: Path = RAW_DIR) -> list[Document]:
    docs: list[Document] = []
    dg_path = raw_dir / "darglobal_pages.json"
    ws_path = raw_dir / "wasalt_pages.json"
    sm_path = raw_dir / "wasalt_sitemap_index.jsonl"

    if dg_path.exists():
        dg_raw = json.loads(dg_path.read_text(encoding="utf-8"))
        docs.extend(darglobal_documents(dg_raw))
        docs.extend(project_directory_documents(dg_raw))
    if ws_path.exists():
        docs.extend(wasalt_documents(json.loads(ws_path.read_text(encoding="utf-8"))))
    if sm_path.exists():
        with sm_path.open(encoding="utf-8") as f:
            docs.append(sitemap_stats_document(json.loads(line) for line in f if line.strip()))

    ids = [d.id for d in docs]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate document ids: {dupes[:5]}")
    return docs


def write_jsonl(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    docs = build(args.raw_dir)
    write_jsonl(docs, args.out)
    counts = Counter(f"{d.source}/{d.kind}" for d in docs)
    log.info("wrote %d documents -> %s", len(docs), args.out)
    for k, n in sorted(counts.items()):
        log.info("  %-24s %d", k, n)


if __name__ == "__main__":
    main()
