import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.retriever import HybridRetriever
from app.rag.service import ChatService
from tests.conftest import FakeLLM


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def make_client(retriever: HybridRetriever):  # type: ignore[no-untyped-def]
    def _make(llm: FakeLLM | None, **overrides) -> TestClient:  # type: ignore[no-untyped-def]
        settings = Settings(_env_file=None, openrouter_api_key=None, **overrides)
        service = ChatService(retriever, llm, top_k=4)
        return TestClient(create_app(settings, service))

    return _make


def test_health(make_client, fake_llm: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["documents"] == 7
    assert body["llm_configured"] is True
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "x-request-id" in r.headers


def test_index_page_served(make_client, fake_llm: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "Property AI Assistant" in r.text


def test_chat_streams_sources_tokens_done(make_client, fake_llm: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm) as client:
        r = client.post("/api/chat", json={"message": "Tell me about Trump Tower Jeddah"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "done"]
    assert events[0][1]["sources"][0]["url"] == "https://darglobal.co.uk/trump-tower-jeddah"
    assert "".join(d["t"] for e, d in events if e == "token") == "Hello world [1]."
    assert events[-1][1]["model"] == "fake/model:free"

    # the prompt contains the retrieved context and the question
    prompt = fake_llm.calls[0][-1]["content"]
    assert "47-floor" in prompt
    assert prompt.endswith("Question: Tell me about Trump Tower Jeddah")
    assert fake_llm.calls[0][0]["role"] == "system"


def test_history_is_forwarded_and_follow_up_inherits_filters(
    make_client, fake_llm: FakeLLM
) -> None:  # type: ignore[no-untyped-def]
    history = [
        {"role": "user", "content": "villas for sale in Riyadh"},
        {"role": "assistant", "content": "Here are some villas."},
    ]
    with make_client(fake_llm) as client:
        r = client.post("/api/chat", json={"message": "under 2 million?", "history": history})
    sources = parse_sse(r.text)[0][1]["sources"]
    listing_ids = [s["url"] for s in sources if s["kind"] == "listing"]
    assert listing_ids == ["https://wasalt.sa/en/property/sale/x-1"]
    roles = [m["role"] for m in fake_llm.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]


def test_llm_error_becomes_error_event(make_client) -> None:  # type: ignore[no-untyped-def]
    with make_client(FakeLLM(fail="All AI models are busy")) as client:
        r = client.post("/api/chat", json={"message": "hello"})
    events = parse_sse(r.text)
    assert events[-1] == ("error", {"message": "All AI models are busy"})


def test_missing_api_key_reports_configuration_error(make_client) -> None:  # type: ignore[no-untyped-def]
    with make_client(None) as client:
        assert client.get("/api/health").json()["llm_configured"] is False
        events = parse_sse(client.post("/api/chat", json={"message": "hello"}).text)
    assert events[-1][0] == "error"
    assert "OPENROUTER_API_KEY" in events[-1][1]["message"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": ""}, {"message": "hi", "history": [{"role": "system", "content": "x"}]}],
)
def test_validation_errors(make_client, fake_llm: FakeLLM, payload: dict) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm) as client:
        assert client.post("/api/chat", json=payload).status_code == 422


def test_message_length_limit(make_client, fake_llm: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm, max_message_chars=20) as client:
        r = client.post("/api/chat", json={"message": "x" * 21})
    assert r.status_code == 422
    assert "too long" in r.json()["error"]


def test_rate_limit(make_client, fake_llm: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm, rate_limit_per_minute=2) as client:
        codes = [client.post("/api/chat", json={"message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_llm_closed_on_shutdown(make_client, fake_llm: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm):
        pass
    assert fake_llm.closed


@pytest.mark.parametrize("path", ["/", "/api/health"])
def test_head_requests_are_supported(make_client, fake_llm: FakeLLM, path: str) -> None:  # type: ignore[no-untyped-def]
    # Uptime monitors probe with HEAD; a 405 would be reported as downtime.
    with make_client(fake_llm) as client:
        get = client.get(path)
        head = client.head(path)
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-type"] == get.headers["content-type"]


def test_docs_redirects_to_api_docs(make_client, fake_llm: FakeLLM) -> None:  # type: ignore[no-untyped-def]
    with make_client(fake_llm) as client:
        r = client.get("/docs", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/api/docs"
        assert client.get("/docs").status_code == 200  # lands on Swagger UI
