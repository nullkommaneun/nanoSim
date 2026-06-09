"""Tests für den LlamaRouter — ohne echten Ollama-Server."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from nanosim.llm.router import LlamaRouter
from nanosim.models import AgentAction, ActionType


@pytest.fixture
def router():
    return LlamaRouter(model="llama3")


class TestDefaultModel:
    """Router-Default und CLI-Default müssen aus derselben Quelle kommen."""

    def test_router_uses_shared_default(self):
        from nanosim.llm.router import DEFAULT_MODEL

        assert LlamaRouter().model == DEFAULT_MODEL

    def test_cli_default_matches_router_default(self):
        from nanosim.llm.router import DEFAULT_MODEL
        from nanosim.main import build_parser

        parser = build_parser()
        assert parser.get_default("model") == DEFAULT_MODEL


class TestCliPersistenceArgs:
    """CLI unterstützt --save und --load für den Weltzustand."""

    def test_save_and_load_args_parsed(self):
        from nanosim.main import build_parser

        args = build_parser().parse_args(["--save", "out.json", "--load", "in.json"])
        assert args.save == "out.json"
        assert args.load == "in.json"

    def test_save_and_load_default_none(self):
        from nanosim.main import build_parser

        args = build_parser().parse_args([])
        assert args.save is None
        assert args.load is None

    def test_trace_arg_parsed(self):
        from nanosim.main import build_parser

        args = build_parser().parse_args(["--trace", "run.jsonl"])
        assert args.trace == "run.jsonl"
        assert build_parser().parse_args([]).trace is None

    def test_replay_arg_parsed(self):
        from nanosim.main import build_parser

        args = build_parser().parse_args(["--replay", "run.jsonl"])
        assert args.replay == "run.jsonl"
        assert build_parser().parse_args([]).replay is None

    def test_report_arg_parsed(self):
        from nanosim.main import build_parser

        args = build_parser().parse_args(
            ["--replay", "run.jsonl", "--report", "out.html"]
        )
        assert args.report == "out.html"
        assert build_parser().parse_args([]).report is None

    def test_dialogue_model_arg_parsed(self):
        from nanosim.main import build_parser

        args = build_parser().parse_args(["--dialogue-model", "llama3.1:8b"])
        assert args.dialogue_model == "llama3.1:8b"
        assert build_parser().parse_args([]).dialogue_model is None

    def test_prompt_variant_and_memory_window_args(self):
        from nanosim.main import build_parser

        args = build_parser().parse_args(
            ["--prompt-variant", "hard", "--memory-window", "6"]
        )
        assert args.prompt_variant == "hard"
        assert args.memory_window == 6
        defaults = build_parser().parse_args([])
        assert defaults.prompt_variant == "baseline"
        assert defaults.memory_window == 3


class TestBuildRouter:
    def test_single_model_returns_llama_router(self):
        from nanosim.llm.router import LlamaRouter
        from nanosim.main import _build_router

        r = _build_router("m", None, "http://localhost:11434")
        assert isinstance(r, LlamaRouter)

    def test_dialogue_model_returns_two_tier(self):
        from nanosim.llm.tiered import TwoTierRouter
        from nanosim.main import _build_router

        r = _build_router("small", "big", "http://localhost:11434")
        assert isinstance(r, TwoTierRouter)
        assert r._decider.model == "small"
        assert r._talker.model == "big"


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
        raw = (
            'Sure! Here you go:\n{"action": "speak", "message": "hi"}\nHope this helps!'
        )
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

        with patch.object(
            router._client, "chat", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await router.think("Was machst du?", AgentAction)

        assert result is not None
        assert result.action == ActionType.IDLE

    @pytest.mark.asyncio
    async def test_passes_json_schema_as_format(self, router: LlamaRouter):
        """think() erzwingt strukturiertes JSON via Ollamas format-Parameter."""
        mock_response = {"message": {"content": '{"action": "idle"}'}}
        mock_chat = AsyncMock(return_value=mock_response)

        with patch.object(router._client, "chat", mock_chat):
            await router.think("Was machst du?", AgentAction)

        fmt = mock_chat.call_args.kwargs["format"]
        assert fmt == AgentAction.model_json_schema()

    @pytest.mark.asyncio
    async def test_retry_also_uses_format(self, router: LlamaRouter):
        """Auch der Retry erzwingt das Schema."""
        bad = {"message": {"content": "kaputt"}}
        good = {"message": {"content": '{"action": "idle"}'}}
        mock_chat = AsyncMock(side_effect=[bad, good])

        with patch.object(router._client, "chat", mock_chat):
            await router.think("Was machst du?", AgentAction)

        assert mock_chat.call_count == 2
        for call in mock_chat.call_args_list:
            assert call.kwargs["format"] == AgentAction.model_json_schema()

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

        with patch.object(
            router._client, "chat", new_callable=AsyncMock, return_value=bad_response
        ):
            result = await router.think("Was machst du?", AgentAction)

        assert result is None

    @pytest.mark.asyncio
    async def test_system_prompt_passed(self, router: LlamaRouter):
        """System-Prompt wird korrekt in die Messages eingefügt."""
        mock_response = {"message": {"content": '{"action": "idle"}'}}
        mock_chat = AsyncMock(return_value=mock_response)

        with patch.object(router._client, "chat", mock_chat):
            await router.think(
                "Was machst du?", AgentAction, system="Du bist eine Katze"
            )

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
