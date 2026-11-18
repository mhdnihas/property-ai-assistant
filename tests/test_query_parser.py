import pytest

from app.rag.query_parser import parse_query


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Villas for sale in Riyadh under 2 million",
            {"city": "Riyadh", "purpose": "sale", "property_type": "Villa", "max_price": 2_000_000},
        ),
        (
            "3 bedroom apartment for rent in Jeddah",
            {"city": "Jeddah", "purpose": "rent", "property_type": "Apartment", "bedrooms": 3},
        ),
        (
            "flats to rent in Mecca below 40k",
            {
                "city": "Makkah",
                "purpose": "rent",
                "property_type": "Apartment",
                "max_price": 40_000,
            },
        ),
        (
            "land in al khobar between 1m and 2.5 million",
            {
                "city": "Khobar",
                "property_type": "Land",
                "min_price": 1_000_000,
                "max_price": 2_500_000,
            },
        ),
        ("4+ bedroom villas in Medina", {"city": "Madinah", "min_bedrooms": 4, "bedrooms": None}),
        ("at least 5 bedrooms in dammam", {"city": "Dammam", "min_bedrooms": 5}),
        (
            "budget of SAR 750,000 for an apartment",
            {"max_price": 750_000, "property_type": "Apartment"},
        ),
    ],
)
def test_parse_extracts_filters(text: str, expected: dict) -> None:
    f = parse_query(text)
    for key, value in expected.items():
        assert getattr(f, key) == value, key


def test_project_question_has_no_listing_intent() -> None:
    f = parse_query("What amenities does Da Vinci Tower have?")
    assert not f.has_constraints
    assert not f.listing_intent


def test_city_plus_type_implies_listing_intent() -> None:
    assert parse_query("villas in Jeddah").listing_intent


def test_small_from_number_is_not_a_price() -> None:
    assert parse_query("apartments from 3 bedrooms").min_price is None


def test_matches_respects_every_constraint() -> None:
    f = parse_query("villas for sale in Riyadh under 2 million")
    ok = {
        "city": "Riyadh",
        "purpose": "sale",
        "property_type": "Villa",
        "price": 1_500_000,
        "bedrooms": 4,
    }
    assert f.matches(ok)
    assert not f.matches({**ok, "price": 2_500_000})
    assert not f.matches({**ok, "city": "Jeddah"})
    assert not f.matches({**ok, "price": None})
