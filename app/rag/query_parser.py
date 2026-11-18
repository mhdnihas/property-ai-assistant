"""Extract structured listing filters (city, purpose, type, bedrooms, price) from a question.

A rule-based parser is deliberate: it is deterministic, testable, free, and
fast. The LLM never decides which listings exist; it only phrases the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.vocab import CITY_PATTERNS, TYPE_PATTERNS

_RENT = re.compile(r"\b(rent|rental|renting|lease|leasing|to let|per month|monthly|yearly)\b", re.I)
_SALE = re.compile(r"\b(buy|buying|sale|sell|selling|purchase|purchasing|own|invest in)\b", re.I)
_BEDS = re.compile(r"\b(\d{1,2})\s*(?:\+\s*)?(?:-?\s*)(?:bed(?:room)?s?|br|bhk)\b", re.I)
_BEDS_MIN = re.compile(r"\b(?:at least|min(?:imum)?|minimum of)\s+(\d{1,2})\s*bed", re.I)
_LISTING_HINT = re.compile(
    r"\b(listing|listings|available|for sale|for rent|show me|find me|options?|cheapest|"
    r"under|below|budget|price|cost|how much)\b",
    re.I,
)

_NUM = r"(\d+(?:[.,]\d+)*)\s*(k|m|mn|million|thousand)?"
_MAX_PRICE = re.compile(
    r"\b(?:under|below|less than|max(?:imum)?|up to|upto|within|budget(?: of| is)?|<)\s*"
    r"(?:sar|sr|riyals?)?\s*" + _NUM,
    re.I,
)
_MIN_PRICE = re.compile(
    r"\b(?:over|above|more than|min(?:imum)?|at least|from|>)\s*(?:sar|sr|riyals?)?\s*" + _NUM, re.I
)
_BETWEEN = re.compile(
    r"\bbetween\s*(?:sar|sr)?\s*" + _NUM + r"\s*(?:and|-|to)\s*(?:sar|sr)?\s*" + _NUM, re.I
)


def _to_number(value: str, suffix: str | None) -> float:
    n = float(value.replace(",", ""))
    s = (suffix or "").lower()
    if s in {"k", "thousand"}:
        n *= 1_000
    elif s in {"m", "mn", "million"}:
        n *= 1_000_000
    return n


@dataclass(slots=True)
class QueryFilters:
    city: str | None = None
    purpose: str | None = None  # "sale" | "rent"
    property_type: str | None = None
    bedrooms: int | None = None
    min_bedrooms: int | None = None
    min_price: float | None = None
    max_price: float | None = None
    listing_intent: bool = False

    @property
    def has_constraints(self) -> bool:
        return any(
            v is not None
            for v in (
                self.city,
                self.purpose,
                self.property_type,
                self.bedrooms,
                self.min_bedrooms,
                self.min_price,
                self.max_price,
            )
        )

    def matches(self, meta: dict) -> bool:
        """True if a listing's metadata satisfies every constraint that was given."""
        if self.city and meta.get("city") != self.city:
            return False
        if self.purpose and meta.get("purpose") != self.purpose:
            return False
        if self.property_type and meta.get("property_type") != self.property_type:
            return False
        beds = meta.get("bedrooms")
        if self.bedrooms is not None and beds != self.bedrooms:
            return False
        if self.min_bedrooms is not None and (beds is None or beds < self.min_bedrooms):
            return False
        price = meta.get("price")
        if self.max_price is not None and (price is None or price > self.max_price):
            return False
        return not (self.min_price is not None and (price is None or price < self.min_price))


def parse_query(text: str) -> QueryFilters:
    f = QueryFilters()

    for city, pattern in CITY_PATTERNS.items():
        if pattern.search(text):
            f.city = city
            break
    for ptype, pattern in TYPE_PATTERNS.items():
        if pattern.search(text):
            f.property_type = ptype
            break

    if _RENT.search(text):
        f.purpose = "rent"
    elif _SALE.search(text):
        f.purpose = "sale"

    if m := _BEDS_MIN.search(text):
        f.min_bedrooms = int(m.group(1))
    elif m := _BEDS.search(text):
        if "+" in m.group(0):
            f.min_bedrooms = int(m.group(1))
        else:
            f.bedrooms = int(m.group(1))

    if m := _BETWEEN.search(text):
        lo = _to_number(m.group(1), m.group(2))
        hi = _to_number(m.group(3), m.group(4))
        f.min_price, f.max_price = min(lo, hi), max(lo, hi)
    else:
        if m := _MAX_PRICE.search(text):
            f.max_price = _to_number(m.group(1), m.group(2))
        if m := _MIN_PRICE.search(text):
            value = _to_number(m.group(1), m.group(2))
            # "from 3 bedrooms" etc. is not a price; ignore implausibly small numbers
            if value >= 1_000:
                f.min_price = value

    # Saudi-city + property-type questions are about Wasalt listings, even without
    # an explicit "show me".
    f.listing_intent = bool(_LISTING_HINT.search(text)) or (
        f.city is not None and (f.property_type is not None or f.purpose is not None)
    )
    return f
