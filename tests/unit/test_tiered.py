"""Tests für den Zwei-Stufen-Router (klein entscheidet, groß spricht)."""

from unittest.mock import AsyncMock

import pytest

from nanosim.llm.tiered import TwoTierRouter
from nanosim.models import ActionType, AgentAction, DecisionAction, SpeechLine


@pytest.fixture
def router():
    return TwoTierRouter(decision_model="small", dialogue_model="big")


class TestTwoTierRouter:
    def test_uses_two_distinct_models(self, router):
        assert router._decider.model == "small"
        assert router._talker.model == "big"

    @pytest.mark.asyncio
    async def test_non_speak_skips_dialogue_model(self, router):
        """Bei move/use/rest entscheidet nur das kleine Modell — das große bleibt still."""
        router._decider.think = AsyncMock(
            return_value=DecisionAction(action=ActionType.MOVE, target="north")
        )
        router._talker.think = AsyncMock()

        result = await router.think("ctx", AgentAction)

        assert result.action == ActionType.MOVE
        assert result.target == "north"
        assert result.message is None
        router._talker.think.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_uses_dialogue_model(self, router):
        """Bei speak formuliert das große Modell die Rede."""
        router._decider.think = AsyncMock(
            return_value=DecisionAction(action=ActionType.SPEAK)
        )
        router._talker.think = AsyncMock(return_value=SpeechLine(message="Miau!"))

        result = await router.think("ctx", AgentAction)

        assert result.action == ActionType.SPEAK
        assert result.message == "Miau!"
        router._talker.think.assert_called_once()

    @pytest.mark.asyncio
    async def test_decider_none_returns_none(self, router):
        """Versagt das kleine Modell, gibt es None — das große wird nicht bemüht."""
        router._decider.think = AsyncMock(return_value=None)
        router._talker.think = AsyncMock()

        result = await router.think("ctx", AgentAction)

        assert result is None
        router._talker.think.assert_not_called()

    @pytest.mark.asyncio
    async def test_talker_none_yields_speak_without_message(self, router):
        """Versagt das große Modell, bleibt die Aktion 'speak' (ohne Text)."""
        router._decider.think = AsyncMock(
            return_value=DecisionAction(action=ActionType.SPEAK)
        )
        router._talker.think = AsyncMock(return_value=None)

        result = await router.think("ctx", AgentAction)

        assert result.action == ActionType.SPEAK
        assert result.message is None

    @pytest.mark.asyncio
    async def test_interface_matches_agent_usage(self, router):
        """think() akzeptiert die gleichen kwargs wie der LlamaRouter (Drop-in)."""
        router._decider.think = AsyncMock(
            return_value=DecisionAction(action=ActionType.IDLE)
        )
        result = await router.think(
            prompt="ctx", response_model=AgentAction, system="sys"
        )
        assert result.action == ActionType.IDLE
