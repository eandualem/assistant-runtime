"""Representative real SDK cleanup; provider operations never reach the network."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI

from assistant_runtime.services.media._providers import generate_google, generate_openai


async def test_openai_cancellation_closes_owned_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = AsyncOpenAI(api_key="test-key")
    client.images.generate = AsyncMock(side_effect=asyncio.CancelledError())
    try:
        with (
            patch("openai.AsyncOpenAI", return_value=client),
            pytest.raises(asyncio.CancelledError),
        ):
            await generate_openai("test", "test", "1024x1024", "medium")
        assert client.is_closed()
    finally:
        await client.close()


async def test_google_closes_both_owned_transports(monkeypatch):
    from google import genai

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    client = genai.Client(api_key="test-key")
    client.close = MagicMock(wraps=client.close)
    client.aio.aclose = AsyncMock(wraps=client.aio.aclose)
    client.aio.models.generate_images = AsyncMock(
        return_value=SimpleNamespace(
            generated_images=[SimpleNamespace(image=SimpleNamespace(image_bytes=b"image"))]
        )
    )
    try:
        with patch("google.genai.Client", return_value=client):
            await generate_google("test", "test", "1024x1024")
        client.close.assert_called_once()
        client.aio.aclose.assert_awaited_once()
    finally:
        await client.aio.aclose()
        client.close()
