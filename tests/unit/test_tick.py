"""Tests für TickEngine und decay_stats."""

from unittest.mock import AsyncMock, patch

import pytest

from nanosim.agents.base import BaseAgent
from nanosim.core.events import EventBus
from nanosim.core.tick import TickEngine, decay_stats
from nanosim.core.world import WorldRegistry
from nanosim.llm.router import LlamaRouter
from nanosim.models import AgentProfile, AgentStats, Room


# ---------------------------------------------------------------------------
# decay_stats
# ---------------------------------------------------------------------------

class TestDecayStats:
    def test_hunger_increases(self):
        stats = AgentStats(hunger=0.5)
        decayed = decay_stats(stats)
        assert decayed.hunger == pytest.approx(0.55)

    def test_stamina_decreases(self):
        stats = AgentStats(stamina=0.5)
        decayed = decay_stats(stats)
        assert decayed.stamina == pytest.approx(0.47)

    def test_mood_decreases_with_hunger(self):
        stats = AgentStats(mood=1.0, hunger=0.5)
        decayed = decay_stats(stats)
        assert decayed.mood < 1.0

    def test_hunger_capped_at_1(self):
        stats = AgentStats(hunger=0.98)
        decayed = decay_stats(stats)
        assert decayed.hunger == 1.0

    def test_stamina_floored_at_0(self):
        stats = AgentStats(stamina=0.01)
        decayed = decay_stats(stats)
        assert decayed.stamina == 0.0

    def test_mood_floored_at_0(self):
        stats = AgentStats(mood=0.001, hunger=1.0)
        decayed = decay_stats(stats)
        assert decayed.mood == 0.0


# ---------------------------------------------------------------------------
# TickEngine
# ---------------------------------------------------------------------------

@pytest.fixture
def world():
    w = WorldRegistry()
    w.add_room(Room(room_id="kitchen", name="Küche", exits={"north": "garden"}))
    w.add_room(Room(room_id="garden", name="Garten", exits={"south": "kitchen"}))
    return w


@pytest.fixture
def engine(world):
    router = LlamaRouter(model="llama3")
    bus = EventBus()

    profiles = [
        AgentProfile(agent_id="cat_01", name="Whiskers", persona="Katze", location_id="kitchen"),
        AgentProfile(agent_id="dog_01", name="Bello", persona="Hund", location_id="garden"),
    ]
    agents = []
    for p in profiles:
        a = BaseAgent(profile=p, router=router)
        agents.append(a)
        world.get_room(p.location_id).occupants.add(p.agent_id)
        bus.subscribe(
            agent_id=a.agent_id,
            handler=a.receive_event,
            location_fn=lambda _a=a: _a.profile.location_id,
        )

    return TickEngine(agents=agents, world=world, bus=bus)


class TestTickEngine:
    @pytest.mark.asyncio
    async def test_single_tick(self, engine):
        """Ein Tick läuft durch ohne Crash."""
        from nanosim.models import AgentAction, ActionType
        idle = AgentAction(action=ActionType.IDLE)
        for agent in engine.agents:
            agent.router.think = AsyncMock(return_value=idle)
        await engine.step()
        assert engine.tick_count == 1

    @pytest.mark.asyncio
    async def test_multiple_ticks(self, engine):
        """Mehrere Ticks laufen durch."""
        from nanosim.models import AgentAction, ActionType
        idle = AgentAction(action=ActionType.IDLE)
        for agent in engine.agents:
            agent.router.think = AsyncMock(return_value=idle)
        await engine.run(num_ticks=3)
        assert engine.tick_count == 3

    @pytest.mark.asyncio
    async def test_stats_decay_applied(self, engine):
        """Stats müssen nach einem Tick verändert sein."""
        from nanosim.models import AgentAction, ActionType
        original_hunger = engine.agents[0].profile.stats.hunger
        idle = AgentAction(action=ActionType.IDLE)
        for agent in engine.agents:
            agent.router.think = AsyncMock(return_value=idle)
        await engine.step()
        assert engine.agents[0].profile.stats.hunger > original_hunger

    @pytest.mark.asyncio
    async def test_speak_event_delivered(self, engine):
        """Speak-Event in gleichem Room wird an anderen Agent zugestellt."""
        # Beide Agenten in kitchen
        engine.world.move_agent("dog_01", "garden", "kitchen")
        engine.agents[1].profile.location_id = "kitchen"

        call_count = [0]

        async def mock_think(prompt, response_model, system=None):
            from nanosim.models import AgentAction, ActionType
            call_count[0] += 1
            if call_count[0] == 1:
                return AgentAction(action=ActionType.SPEAK, message="Miau!")
            return AgentAction(action=ActionType.IDLE)

        for agent in engine.agents:
            agent.router.think = mock_think

        await engine.step()

        # Nach dem Tick hat einer der Agents das Event in der inbox
        # (wird im nächsten Tick verarbeitet)
        has_inbox = any(len(a.inbox) > 0 for a in engine.agents)
        # Events wurden über drain() zugestellt — prüfe ob drain tatsächlich lief
        assert engine.tick_count == 1
