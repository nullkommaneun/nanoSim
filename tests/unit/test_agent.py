"""Tests für BaseAgent und Prompt-Builder."""

from unittest.mock import AsyncMock, patch

import pytest

from nanosim.agents.base import BaseAgent
from nanosim.agents.prompt import build_prompt, build_system_prompt
from nanosim.core.world import WorldRegistry
from nanosim.llm.router import LlamaRouter
from nanosim.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    AgentStats,
    BaseEvent,
    EventType,
    Room,
)


@pytest.fixture
def world():
    w = WorldRegistry()
    w.add_room(Room(
        room_id="kitchen", name="Küche",
        description="Eine warme Küche.",
        objects=["futternapf"],
        exits={"north": "garden"},
    ))
    w.add_room(Room(
        room_id="garden", name="Garten",
        description="Ein sonniger Garten.",
        objects=["gras"],
        exits={"south": "kitchen"},
    ))
    return w


@pytest.fixture
def cat_profile():
    return AgentProfile(
        agent_id="cat_01", name="Whiskers",
        persona="Neugierige Katze.",
        location_id="kitchen",
        stats=AgentStats(stamina=0.8, mood=0.9, hunger=0.3),
    )


@pytest.fixture
def router():
    return LlamaRouter(model="llama3")


@pytest.fixture
def agent(cat_profile, router):
    return BaseAgent(profile=cat_profile, router=router)


# ---------------------------------------------------------------------------
# Prompt-Builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_build_prompt_contains_location(self, cat_profile, world):
        world.get_room("kitchen").occupants.add("cat_01")
        prompt = build_prompt(cat_profile, world)
        assert "Küche" in prompt
        assert "warme Küche" in prompt

    def test_build_prompt_contains_stats(self, cat_profile, world):
        prompt = build_prompt(cat_profile, world)
        assert "Energie=0.8" in prompt
        assert "Hunger=0.3" in prompt

    def test_build_prompt_contains_objects(self, cat_profile, world):
        prompt = build_prompt(cat_profile, world)
        assert "futternapf" in prompt

    def test_build_prompt_contains_exits(self, cat_profile, world):
        prompt = build_prompt(cat_profile, world)
        assert "north → garden" in prompt

    def test_build_prompt_shows_others(self, cat_profile, world):
        world.get_room("kitchen").occupants.add("cat_01")
        world.get_room("kitchen").occupants.add("dog_01")
        prompt = build_prompt(cat_profile, world)
        assert "dog_01" in prompt

    def test_build_prompt_no_memory(self, cat_profile, world):
        prompt = build_prompt(cat_profile, world)
        assert "keine" in prompt  # Erinnerungen: keine

    def test_build_prompt_with_memory(self, cat_profile, world):
        cat_profile.add_memory("Tick 1: Habe Milch getrunken")
        prompt = build_prompt(cat_profile, world)
        assert "Milch" in prompt

    def test_build_system_prompt(self, cat_profile):
        system = build_system_prompt(cat_profile)
        assert "Whiskers" in system
        assert "Katze" in system
        assert "Deutsch" in system


# ---------------------------------------------------------------------------
# BaseAgent — Event-Handling
# ---------------------------------------------------------------------------

class TestAgentInbox:
    @pytest.mark.asyncio
    async def test_receive_event(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_SPEAK, source="dog_01",
            payload={"message": "Wuff!"},
        )
        await agent.receive_event(event)
        assert len(agent.inbox) == 1

    def test_process_inbox_speak(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_SPEAK, source="dog_01",
            payload={"message": "Wuff!"},
        )
        agent.inbox.append(event)
        agent.process_inbox(tick=5)
        assert len(agent.inbox) == 0
        assert len(agent.profile.memory) == 1
        assert "Wuff!" in agent.profile.memory[0]
        assert "Tick 5" in agent.profile.memory[0]

    def test_process_inbox_move(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_MOVE, source="dog_01",
            payload={"from": "garden", "to": "kitchen"},
        )
        agent.inbox.append(event)
        agent.process_inbox(tick=3)
        assert "kitchen" in agent.profile.memory[0]

    def test_process_inbox_use(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_USE, source="dog_01",
            payload={"object": "ball"},
        )
        agent.inbox.append(event)
        agent.process_inbox(tick=1)
        assert "ball" in agent.profile.memory[0]

    def test_process_inbox_rest(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_REST, source="dog_01",
        )
        agent.inbox.append(event)
        agent.process_inbox(tick=2)
        assert "ruhte" in agent.profile.memory[0].lower()


# ---------------------------------------------------------------------------
# BaseAgent — Action Execution
# ---------------------------------------------------------------------------

class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_tick_speak(self, agent, world):
        world.get_room("kitchen").occupants.add("cat_01")
        mock_resp = {"message": {"content": '{"action": "speak", "message": "Miau!"}'}}
        with patch.object(agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_SPEAK
        assert event.payload["message"] == "Miau!"
        assert "Miau!" in agent.profile.memory[-1]

    @pytest.mark.asyncio
    async def test_tick_move(self, agent, world):
        world.get_room("kitchen").occupants.add("cat_01")
        mock_resp = {"message": {"content": '{"action": "move", "target": "north"}'}}
        with patch.object(agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_MOVE
        assert agent.profile.location_id == "garden"
        assert "cat_01" not in world.agents_in_room("kitchen")
        assert "cat_01" in world.agents_in_room("garden")

    @pytest.mark.asyncio
    async def test_tick_move_invalid_exit(self, agent, world):
        world.get_room("kitchen").occupants.add("cat_01")
        mock_resp = {"message": {"content": '{"action": "move", "target": "west"}'}}
        with patch.object(agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            event = await agent.tick(world, tick=0)
        assert event is None
        assert agent.profile.location_id == "kitchen"  # Nicht bewegt

    @pytest.mark.asyncio
    async def test_tick_use(self, agent, world):
        world.get_room("kitchen").occupants.add("cat_01")
        mock_resp = {"message": {"content": '{"action": "use", "target": "futternapf"}'}}
        with patch.object(agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_USE
        assert event.payload["object"] == "futternapf"

    @pytest.mark.asyncio
    async def test_tick_rest(self, agent, world):
        old_stamina = agent.profile.stats.stamina
        mock_resp = {"message": {"content": '{"action": "rest"}'}}
        with patch.object(agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_REST
        assert agent.profile.stats.stamina > old_stamina

    @pytest.mark.asyncio
    async def test_tick_idle(self, agent, world):
        mock_resp = {"message": {"content": '{"action": "idle"}'}}
        with patch.object(agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            event = await agent.tick(world, tick=0)
        assert event is None

    @pytest.mark.asyncio
    async def test_tick_llm_failure_fallback_idle(self, agent, world):
        """Wenn LLM komplett versagt → idle, kein Crash."""
        mock_resp = {"message": {"content": "Ich bin kaputt"}}
        with patch.object(agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp):
            event = await agent.tick(world, tick=0)
        assert event is None  # idle → kein Event
