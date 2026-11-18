import json

import httpx
import pytest

from app.llm.openrouter import LLMError, OpenRouterClient


def sse(*contents: str, error: str | None = None) -> bytes:
    lines = [": OPENROUTER PROCESSING", ""]
    for c in contents:
        lines += [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}", ""]
    if error:
        lines += [f"data: {json.dumps({'error': {'message': error}})}", ""]
    lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def client_with(handler) -> OpenRouterClient:  # type: ignore[no-untyped-def]
    return OpenRouterClient(
        "test-key", ["a:free", "b:free"], transport=httpx.MockTransport(handler)
    )


async def collect(client: OpenRouterClient) -> list[tuple[str, str]]:
    return [
        (d.text, d.model) async for d in client.stream_chat([{"role": "user", "content": "hi"}])
    ]


async def test_streams_tokens_and_sends_auth_header() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse("Hel", "lo"))

    client = client_with(handler)
    assert await collect(client) == [("Hel", "a:free"), ("lo", "a:free")]
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["stream"] is True
    assert seen["body"]["model"] == "a:free"
    await client.aclose()


async def test_falls_back_to_next_model_on_429() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == "a:free":
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, content=sse("ok"))

    assert await collect(client_with(handler)) == [("ok", "b:free")]


async def test_falls_back_on_midstream_error_before_first_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(
            200, content=sse(error="overloaded") if model == "a:free" else sse("fine")
        )

    assert await collect(client_with(handler)) == [("fine", "b:free")]


async def test_error_after_first_token_is_not_retried() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse("partial", error="boom"))

    client = client_with(handler)
    with pytest.raises(LLMError, match="interrupted"):
        await collect(client)


async def test_all_models_failing_raises_friendly_error() -> None:
    client = client_with(lambda r: httpx.Response(503, text="down"))
    with pytest.raises(LLMError, match="busy"):
        await collect(client)


async def test_bad_key_fails_fast() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": "nope"})

    with pytest.raises(LLMError, match="API key"):
        await collect(client_with(handler))
    assert len(calls) == 1  # no pointless retries on auth errors


async def test_network_error_falls_back() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "a:free":
            raise httpx.ConnectTimeout("slow")
        return httpx.Response(200, content=sse("ok"))

    assert await collect(client_with(handler)) == [("ok", "b:free")]
