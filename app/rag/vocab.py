"""Normalisation vocabulary shared by the data pipeline and the query parser.

Keeping a single source of truth guarantees that a city/type written one way in
the scraped data ("Aldammam") matches the same city in a user question ("Dammam").
"""

from __future__ import annotations

import re

# canonical city -> aliases (lower-case) found in data or in user questions
CITY_ALIASES: dict[str, tuple[str, ...]] = {
    "Riyadh": ("riyadh", "riyad", "ar riyadh"),
    "Jeddah": ("jeddah", "jedda", "jiddah", "jeddha"),
    "Dammam": ("dammam", "aldammam", "al dammam", "ad dammam"),
    "Khobar": ("khobar", "al khobar", "al-khobar", "alkhobar"),
    "Makkah": ("makkah", "mecca", "makkah al mukarramah", "makka"),
    "Madinah": ("madinah", "medina", "madina", "al madinah"),
}

# canonical property type -> aliases (lower-case, singular + plural)
TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "Villa": ("villa", "villas", "townhouse", "townhouses"),
    "Apartment": ("apartment", "apartments", "flat", "flats"),
    "Land": ("land", "lands", "plot", "plots"),
    "Building": ("building", "buildings"),
    "Floor": ("floor", "floors"),
}


def _alias_pattern(aliases: tuple[str, ...]) -> re.Pattern[str]:
    alts = sorted((re.escape(a).replace(r"\ ", r"[\s-]+") for a in aliases), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)


CITY_PATTERNS = {city: _alias_pattern(a) for city, a in CITY_ALIASES.items()}
TYPE_PATTERNS = {ptype: _alias_pattern(a) for ptype, a in TYPE_ALIASES.items()}


def normalize_city(raw: str | None) -> str | None:
    """Map a raw city string from the data to its canonical name, if known."""
    if not raw:
        return None
    for city, pattern in CITY_PATTERNS.items():
        if pattern.search(raw):
            return city
    return raw.strip() or None


def normalize_type(raw: str | None) -> str | None:
    if not raw:
        return None
    for ptype, pattern in TYPE_PATTERNS.items():
        if pattern.search(raw):
            return ptype
    return raw.strip() or None
