from pipeline.build_documents import (
    chunk_segments,
    darglobal_documents,
    listing_document,
    sitemap_stats_document,
    wasalt_documents,
)

LONG = "word " * 100


def test_chunk_segments_respects_max_size() -> None:
    segments = [LONG.strip()] * 10
    chunks = chunk_segments(segments, max_chars=1200)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)
    assert " ".join(chunks).count("word") >= 1000  # nothing lost


def test_chunk_segments_splits_oversized_segment_on_sentences() -> None:
    text = "This is a sentence. " * 200
    chunks = chunk_segments([text], max_chars=500)
    assert all(len(c) <= 500 for c in chunks)


def test_darglobal_skips_errors_short_pages_and_duplicates() -> None:
    text = "Trump Tower Jeddah | " + "A seafront tower with Red Sea views. " * 10
    raw = {
        "pages": [
            {
                "path": "/a",
                "url": "https://darglobal.co.uk/a",
                "status": 200,
                "title": "A | DarGlobal",
                "text": text,
            },
            {
                "path": "/b",
                "url": "https://darglobal.co.uk/b",
                "status": 200,
                "title": "B",
                "text": text,
            },  # dup
            {
                "path": "/c",
                "url": "https://darglobal.co.uk/c",
                "status": 500,
                "title": "C",
                "text": "",
            },
            {
                "path": "/d",
                "url": "https://darglobal.co.uk/d",
                "status": 200,
                "title": "D",
                "text": "tiny",
            },
            {
                "path": "/projects",
                "url": "https://darglobal.co.uk/projects",
                "status": 200,
                "title": "Projects",
                "text": "Projects | Oman | Spain | " + "x " * 100,
            },
        ]
    }
    docs = list(darglobal_documents(raw))
    assert {d.meta["path"] for d in docs} == {"/a", "/projects"}
    assert docs[0].title == "A"  # brand suffix stripped
    assert {d.kind for d in docs} == {"project", "catalogue"}


def test_listing_document_normalises_fields() -> None:
    doc = listing_document(
        {
            "id": 7,
            "title": "Villa 450 SQM",
            "url": "https://wasalt.sa/en/property/sale/v-7",
            "purpose": "sale",
            "type": "Villa",
            "city": "Aldammam",
            "district": "Al Shatea",
            "address": "Al Shatea, Dammam",
            "price": 2_250_000,
            "currency": "SAR",
            "attributes": {"noOfBedrooms": 5, "carpetArea": "450 SQM", "completionYear": "New"},
            "amenities": ["Park"],
            "rega_licensed": True,
            "published_at": "2026-11-13T10:00:00Z",
        }
    )
    assert doc is not None
    assert doc.meta["city"] == "Dammam"
    assert doc.meta["bedrooms"] == 5
    assert doc.meta["land_area_sqm"] == 450
    assert "2,250,000 SAR" in doc.text
    assert "Price per sqm: 5,000 SAR" in doc.text
    assert "new build" in doc.text


def test_listing_document_skips_placeholders() -> None:
    assert listing_document({"url": None, "attributes": {}}) is None


def test_rent_listing_shows_period() -> None:
    doc = listing_document(
        {
            "id": 9,
            "title": "Apt",
            "url": "https://wasalt.sa/x",
            "purpose": "rent",
            "type": "Apartment",
            "city": "Jeddah",
            "price": 90_000,
            "attributes": {},
            "rent_frequency": {"yearly": {"amount": 90_000, "enCurrencyType": "SAR"}},
        }
    )
    assert doc is not None
    assert "Rent: 90,000 SAR yearly" in doc.text


def test_wasalt_documents_dedupes_listings_and_faqs() -> None:
    listing = {
        "id": 1,
        "title": "Villa",
        "url": "https://wasalt.sa/1",
        "purpose": "sale",
        "type": "Villa",
        "city": "Riyadh",
        "price": 1,
        "attributes": {},
    }
    page = {
        "path": "/en/p",
        "url": "https://wasalt.sa/en/p",
        "title": "Villas",
        "total_count": 10,
        "overview": "Riyadh is great. It has villas.",
        "faq": [{"q": "Q1?", "a": "A1"}],
        "listings": [listing],
    }
    docs = list(wasalt_documents({"pages": [page, {**page, "path": "/en/q"}]}))
    kinds = [d.kind for d in docs]
    assert kinds.count("listing") == 1
    assert kinds.count("faq") == 1
    assert kinds.count("market_guide") == 2


def test_sitemap_stats() -> None:
    recs = [{"purpose": "sale", "type": "villa", "bedrooms": 3}] * 150 + [
        {"purpose": "rent", "type": "apartment"}
    ]
    doc = sitemap_stats_document(recs)
    assert "151 active property listings" in doc.text
    assert "villa: 150" in doc.text
    assert "3 bedrooms: 150" in doc.text


def test_project_directory_groups_projects_by_country() -> None:
    from pipeline.build_documents import project_directory_documents

    text = " | ".join(
        [
            "Projects",
            "find your next investment",
            "Oman",
            "Trump Cliff Villas",
            "Muscat, Oman",
            "Fairway Villas",
            "AIDA",
            "Oman",
            "Amour Sans Detour",  # "Oman" here is a location, not a new heading
            "Spain",
            "Tierra Viva",
            "Benahavís, Spain",
            "More about Prime Locations",
            "long SEO text that must be ignored",
        ]
    )
    raw = {
        "pages": [{"path": "/projects", "url": "https://darglobal.co.uk/projects", "text": text}]
    }
    docs = {d.title: d for d in project_directory_documents(raw)}

    oman = docs["DarGlobal projects in Oman"]
    assert "Trump Cliff Villas" in oman.text
    assert "Amour Sans Detour" in oman.text  # kept in Oman despite the repeated "Oman"
    assert "Tierra Viva" not in oman.text
    assert oman.url == "https://darglobal.co.uk/projects/properties-in-oman"
    assert "Tierra Viva" in docs["DarGlobal projects in Spain"].text

    overview = docs["DarGlobal project directory (all countries)"]
    assert "2 countries: Oman, Spain" in overview.text
    assert "SEO" not in overview.text


def test_project_directory_absent_page_yields_nothing() -> None:
    from pipeline.build_documents import project_directory_documents

    assert list(project_directory_documents({"pages": []})) == []
