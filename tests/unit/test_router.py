"""Tests für den LlamaRouter — ohne echten Ollama-Server."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from nanosim.llm.router import LlamaRouter
from nanosim.models import AgentAction, ActionType


@pytest.fixture
def router():
    return LlamaRouter(model="llama3")


class TestJsonExtraction:
    """Tests für _extract_json (statische Methode, kein Ollama nötig)."""

    def test_clean_json(self):
        raw = '{"action": "idle"}'
        assert LlamaRouter._extract_json(raw) == '{"action": "idle"}'

    def test_json_with_markdown_block(self):
        raw = 'Here is the result:\n```json\n{"action": "idle"}\n```'
        assert LlamaRouter._extract_json(raw) == '{"action": "idle"}'

    def test_json_with_generic_codeblock(self):
        raw = '```\n{"action": "idle"}\n```'
        assert LlamaRouter._extract_json(raw) == '{"action": "idle"}'

    def test_json_with_surrounding_text(self):
        raw = 'Sure! Here you go:\n{"action": "speak", "message": "hi"}\nHope this helps!'
        result = LlamaRouter._extract_json(raw)
        parsed = json.loads(result)
        assert parsed["action"] == "speak"

    def test_no_json_returns_stripped(self):
        raw = "   no json here   "
        assert LlamaRouter._extract_json(raw) == "no json here"


class TestParseAndValidate:
    def test_valid_json(self):
        raw = '{"action": "idle"}'
        result = LlamaRouter._parse_and_validate(raw, AgentAction)
        assert result is not None
        assert result.action == ActionType.IDLE

    def test_valid_with_optional_fields(self):
        raw = '{"action": "speak", "message": "Miau!", "target": null}'
        result = LlamaRouter._parse_and_validate(raw, AgentAction)
        assert result.message == "Miau!"

    def test_invalid_json(self):
        raw = "not json at all"
        result = LlamaRouter._parse_and_validate(raw, AgentAction)
        assert result is None

    def test_valid_json_wrong_schema(self):
        raw = '{"wrong_field": 123}'
        result = LlamaRouter._parse_and_validate(raw, AgentAction)
        assert result is None

    def test_json_in_markdown(self):
        raw = '```json\n{"action": "rest"}\n```'
        result = LlamaRouter._parse_and_validate(raw, AgentAction)
        assert result is not None
        assert result.action == ActionType.REST


class TestThink:
    @pytest.mark.asyncio
    async def test_successful_first_attempt(self, router: LlamaRouter):
        """Erster Versuch liefert valides JSON → kein Retry."""
        mock_response = {"message": {"content": '{"action": "idle"}'}}

        with patch.object(router._client, "chat", new_callable=AsyncMock, return_value=mock_response):
            result = await router.think("Was machst du?", AgentAction)

        assert result is not None
        assert result.action == ActionType.IDLE

    @pytest.mark.asyncio
    async def test_retry_on_bad_json(self, router: LlamaRouter):
        """Erster Versuch kaputt → Retry → Erfolg."""
        bad_response = {"message": {"content": "Sure! {action: idle}"}}
        good_response = {"message": {"content": '{"action": "speak", "message": "hi"}'}}

        mock_chat = AsyncMock(side_effect=[bad_response, good_response])

        with patch.object(router._client, "chat", mock_chat):
            result = await router.think("Was machst du?", AgentAction)

        assert result is not None
        assert result.action == ActionType.SPEAK
        assert mock_chat.call_count == 2

    @pytest.mark.asyncio
    async def test_both_attempts_fail(self, router: LlamaRouter):
        """Beide Versuche fehlgeschlagen → None."""
        bad_response = {"message": {"content": "I cannot do JSON"}}

        with patch.object(router._client, "chat", new_callable=AsyncMock, return_value=bad_response):
            result = await router.think("Was machst du?", AgentAction)

        assert result is None

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self, router: LlamaRouter):
        """System-Prompt wird korrekt in die Messages eingefügt."""
        mock_response = {"message": {"content": '{"action": "idle"}'}}
        mock_chat = AsyncMock(return_value=mock_response)

        with patch.object(router._client, "chat", mock_chat):
            await router.think("Was machst du?", AgentAction, system="Du bist eine Katze")

        messages = mock_chat.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert "Katze" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_semaphore_serializes(self, router: LlamaRouter):
        """Semaphore stellt sicher, dass Calls serialisiert werden."""
        call_order: list[int] = []

        async def slow_chat(**kwargs):
            import asyncio
            call_order.append(len(call_order))
            await asyncio.sleep(0.05)
            return {"message": {"content": '{"action": "idle"}'}}

        with patch.object(router._client, "chat", side_effect=slow_chat):
            results = await __import__("asyncio").gather(
                router.think("A", AgentAction),
                router.think("B", AgentAction),
            )

        assert all(r is not None for r in results)
        # Beide Calls müssen durchgegangen sein (sequentiell durch Semaphore)
        assert len(call_order) == 2
