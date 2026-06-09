"""Tests für BaseAgent und Prompt-Builder."""

from unittest.mock import AsyncMock, patch

import pytest

from nanosim.agents.base import BaseAgent
from nanosim.agents.prompt import build_prompt, build_system_prompt
from nanosim.core.world import WorldRegistry
from nanosim.llm.router import LlamaRouter
from nanosim.models import (
    AgentProfile,
    AgentStats,
    BaseEvent,
    EventType,
    Room,
)


@pytest.fixture
def world():
    w = WorldRegistry()
    w.add_room(
        Room(
            room_id="kitchen",
            name="Küche",
            description="Eine warme Küche.",
            objects=["futternapf"],
            exits={"north": "garden"},
        )
    )
    w.add_room(
        Room(
            room_id="garden",
            name="Garten",
            description="Ein sonniger Garten.",
            objects=["gras"],
            exits={"south": "kitchen"},
        )
    )
    return w


@pytest.fixture
def cat_profile():
    return AgentProfile(
        agent_id="cat_01",
        name="Whiskers",
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

    def test_build_prompt_senses_adjacent_other(self, cat_profile, world):
        """Ein Tier im Nachbarraum wird mit Richtung wahrgenommen (Sozial-Gespür)."""
        world.get_room("garden").occupants.add("dog_01")  # garden ist north von kitchen
        prompt = build_prompt(cat_profile, world)
        assert "dog_01" in prompt
        assert "north" in prompt

    def test_build_prompt_senses_distant_other_with_distance(self):
        """Ein erreichbares Tier mehrere Räume entfernt wird mit Entfernung wahrgenommen."""
        from nanosim.world.rooms import create_default_world

        w = create_default_world()
        cat = AgentProfile(
            agent_id="cat_01",
            name="Whiskers",
            persona="Katze",
            location_id="kitchen",
        )
        w.get_room("balcony").occupants.add("parrot_01")  # balcony ist 2 Räume entfernt
        prompt = build_prompt(cat, w)
        assert "parrot_01" in prompt
        assert "2 Räume entfernt" in prompt

    def test_first_step_directions_bfs(self):
        """Breitensuche liefert Entfernung + Richtung des ersten Schritts."""
        from nanosim.agents.prompt import _first_step_directions
        from nanosim.world.rooms import create_default_world

        w = create_default_world()
        d = _first_step_directions(w, "kitchen")
        assert d["garden"] == (1, "north")
        assert d["living_room"] == (1, "east")
        dist, direction = d["balcony"]
        assert dist == 2
        assert direction in ("north", "east")  # erster Schritt über garden/living_room

    def test_build_prompt_ignores_unreachable_other(self):
        """Ein Tier in einem nicht erreichbaren (isolierten) Raum wird nicht wahrgenommen."""
        from nanosim.core.world import WorldRegistry
        from nanosim.models import Room

        w = WorldRegistry()
        w.add_room(Room(room_id="a", name="A", exits={}))
        w.add_room(Room(room_id="b", name="B", exits={}))  # keine Verbindung
        cat = AgentProfile(
            agent_id="cat_01", name="W", persona="Katze", location_id="a"
        )
        w.get_room("b").occupants.add("dog_01")
        prompt = build_prompt(cat, w)
        assert "dog_01" not in prompt

    def test_system_prompt_has_social_drive(self, cat_profile):
        """Der System-Prompt gibt dem Tier einen geselligen Charakter."""
        system = build_system_prompt(cat_profile).lower()
        assert "gesellig" in system or "andere" in system

    def test_system_prompt_emphasizes_staying(self, cat_profile):
        """Der Charakter betont eindeutig: bei Gesellschaft bleiben."""
        system = build_system_prompt(cat_profile).lower()
        assert "bleib bei ihnen" in system

    def test_build_prompt_urges_staying_when_others_present(self, cat_profile, world):
        """Ist jemand im Raum, drängt der Prompt situativ zum Bleiben."""
        world.get_room("kitchen").occupants.add("cat_01")
        world.get_room("kitchen").occupants.add("dog_01")
        prompt = build_prompt(cat_profile, world)
        assert "Bleib hier" in prompt

    def test_build_prompt_no_stay_urge_when_alone(self, cat_profile, world):
        """Ist das Tier allein, gibt es keinen Bleib-Aufruf."""
        prompt = build_prompt(cat_profile, world)
        assert "Bleib hier" not in prompt

    def test_build_prompt_urges_moving_when_alone_with_neighbor(
        self, cat_profile, world
    ):
        """Allein, aber jemand erreichbar → klarer Aufruf, hinzugehen (mit Richtung)."""
        world.get_room("garden").occupants.add("dog_01")  # garden = north von kitchen
        prompt = build_prompt(cat_profile, world)
        assert "geh zu den anderen" in prompt.lower()
        assert "north" in prompt

    def test_build_prompt_no_move_urge_when_together(self, cat_profile, world):
        """Ist schon jemand im Raum, gibt es keinen Geh-Aufruf (sondern Bleiben)."""
        world.get_room("kitchen").occupants.add("cat_01")
        world.get_room("kitchen").occupants.add("dog_01")
        prompt = build_prompt(cat_profile, world)
        assert "geh zu den anderen" not in prompt.lower()
        assert "Bleib hier" in prompt

    # --- Gedächtnis-Fenster ---

    def test_build_prompt_memory_window_limits(self, cat_profile, world):
        """memory_window begrenzt, wie viele Erinnerungen im Prompt erscheinen."""
        for m in ["E-eins", "E-zwei", "E-drei"]:
            cat_profile.add_memory(m)
        prompt = build_prompt(cat_profile, world, memory_window=1)
        assert "E-drei" in prompt
        assert "E-eins" not in prompt

    def test_build_prompt_memory_window_zero_shows_none(self, cat_profile, world):
        cat_profile.add_memory("E-eins")
        prompt = build_prompt(cat_profile, world, memory_window=0)
        assert "E-eins" not in prompt

    # --- Prompt-Varianten ---

    def test_prompt_variants_has_expected_keys(self):
        from nanosim.agents.prompt import PROMPT_VARIANTS

        assert {"baseline", "soft", "hard", "naming"} <= set(PROMPT_VARIANTS)

    def test_system_prompt_variant_hard_demands_question(self, cat_profile):
        system = build_system_prompt(cat_profile, variant="hard").lower()
        assert "frage" in system

    def test_system_prompt_variant_soft_is_neutral(self, cat_profile):
        system = build_system_prompt(cat_profile, variant="soft").lower()
        assert "nicht weg" not in system

    def test_build_prompt_variant_threads_stay_text(self, cat_profile, world):
        world.get_room("kitchen").occupants.add("cat_01")
        world.get_room("kitchen").occupants.add("dog_01")
        prompt = build_prompt(cat_profile, world, variant="hard")
        assert "BLEIB HIER" in prompt

    def test_default_variant_is_baseline_behaviour(self, cat_profile, world):
        """Ohne Variante bleibt alles wie bisher (verhaltensgleich)."""
        world.get_room("kitchen").occupants.add("cat_01")
        world.get_room("kitchen").occupants.add("dog_01")
        assert "Bleib hier" in build_prompt(cat_profile, world)
        assert "bleib bei ihnen" in build_system_prompt(cat_profile).lower()


class TestAgentConfig:
    def test_agent_stores_prompt_variant_and_memory_window(self, cat_profile, router):
        agent = BaseAgent(
            profile=cat_profile, router=router,
            prompt_variant="hard", memory_window=5,
        )
        assert agent.prompt_variant == "hard"
        assert agent.memory_window == 5

    def test_agent_defaults_are_baseline(self, cat_profile, router):
        agent = BaseAgent(profile=cat_profile, router=router)
        assert agent.prompt_variant == "baseline"
        assert agent.memory_window == 3

    @pytest.mark.asyncio
    async def test_agent_threads_memory_window_into_prompt(self, cat_profile, router, world):
        from nanosim.models import ActionType, AgentAction

        for m in ["alpha", "beta", "gamma"]:
            cat_profile.add_memory(m)
        agent = BaseAgent(profile=cat_profile, router=router, memory_window=1)
        captured = {}

        async def fake_think(prompt, response_model, system=None):
            captured["prompt"] = prompt
            return AgentAction(action=ActionType.IDLE)

        agent.router.think = fake_think
        await agent.tick(world, tick=0)
        assert "gamma" in captured["prompt"]
        assert "alpha" not in captured["prompt"]


# ---------------------------------------------------------------------------
# BaseAgent — Event-Handling
# ---------------------------------------------------------------------------


class TestAgentInbox:
    @pytest.mark.asyncio
    async def test_receive_event(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_SPEAK,
            source="dog_01",
            payload={"message": "Wuff!"},
        )
        await agent.receive_event(event)
        assert len(agent.inbox) == 1

    def test_process_inbox_speak(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_SPEAK,
            source="dog_01",
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
            type=EventType.AGENT_MOVE,
            source="dog_01",
            payload={"from": "garden", "to": "kitchen"},
        )
        agent.inbox.append(event)
        agent.process_inbox(tick=3)
        assert "kitchen" in agent.profile.memory[0]

    def test_process_inbox_use(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_USE,
            source="dog_01",
            payload={"object": "ball"},
        )
        agent.inbox.append(event)
        agent.process_inbox(tick=1)
        assert "ball" in agent.profile.memory[0]

    def test_process_inbox_rest(self, agent):
        event = BaseEvent(
            type=EventType.AGENT_REST,
            source="dog_01",
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
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_SPEAK
        assert event.payload["message"] == "Miau!"
        assert "Miau!" in agent.profile.memory[-1]

    @pytest.mark.asyncio
    async def test_tick_move(self, agent, world):
        world.get_room("kitchen").occupants.add("cat_01")
        mock_resp = {"message": {"content": '{"action": "move", "target": "north"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
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
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is None
        assert agent.profile.location_id == "kitchen"  # Nicht bewegt

    @pytest.mark.asyncio
    async def test_tick_use(self, agent, world):
        world.get_room("kitchen").occupants.add("cat_01")
        mock_resp = {
            "message": {"content": '{"action": "use", "target": "futternapf"}'}
        }
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_USE
        assert event.payload["object"] == "futternapf"

    @pytest.mark.asyncio
    async def test_tick_use_applies_object_effect(self, agent, world):
        """Ein vorhandenes Objekt benutzen verändert die Stats (futternapf → Hunger runter)."""
        world.get_room("kitchen").occupants.add("cat_01")
        agent.profile.stats = AgentStats(stamina=0.8, mood=0.8, hunger=0.9)
        mock_resp = {
            "message": {"content": '{"action": "use", "target": "futternapf"}'}
        }
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_USE
        assert agent.profile.stats.hunger < 0.9  # gefressen → weniger Hunger

    @pytest.mark.asyncio
    async def test_tick_use_absent_object_has_no_effect(self, agent, world):
        """Ein Objekt benutzen, das nicht im Raum ist → kein Event, keine Wirkung."""
        world.get_room("kitchen").occupants.add("cat_01")
        before = agent.profile.stats.model_copy()
        mock_resp = {
            "message": {"content": '{"action": "use", "target": "sofa"}'}
        }  # sofa nicht in kitchen
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is None
        assert agent.profile.stats == before  # Stats unverändert

    @pytest.mark.asyncio
    async def test_tick_rest(self, agent, world):
        old_stamina = agent.profile.stats.stamina
        mock_resp = {"message": {"content": '{"action": "rest"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.type == EventType.AGENT_REST
        assert agent.profile.stats.stamina > old_stamina

    @pytest.mark.asyncio
    async def test_tick_idle(self, agent, world):
        mock_resp = {"message": {"content": '{"action": "idle"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is None

    @pytest.mark.asyncio
    async def test_tick_llm_failure_fallback_idle(self, agent, world):
        """Wenn LLM komplett versagt → idle, kein Crash."""
        mock_resp = {"message": {"content": "Ich bin kaputt"}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is None  # idle → kein Event

    @pytest.mark.asyncio
    async def test_tick_speak_without_message_becomes_idle(self, agent, world):
        """speak ohne Text → kein Event, kein Geister-Speak (wie idle)."""
        world.get_room("kitchen").occupants.add("cat_01")
        mock_resp = {"message": {"content": '{"action": "speak", "message": null}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is None
        assert not any("sagte: 'None'" in m for m in agent.profile.memory)

    def test_execute_speak_empty_string_no_event(self, agent, world):
        """Reiner Whitespace als Rede → ebenfalls kein Event."""
        from nanosim.models import ActionType, AgentAction

        action = AgentAction(action=ActionType.SPEAK, message="   ")
        event = agent._execute(action, world, tick=0)
        assert event is None


# ---------------------------------------------------------------------------
# BaseAgent — Causality-Depth (Schutz vor Endlos-Event-Ketten)
# ---------------------------------------------------------------------------


class TestCausalityDepth:
    @pytest.mark.asyncio
    async def test_spontaneous_speak_has_depth_zero(self, agent, world):
        """Wer nichts gehört hat, beginnt eine frische Kette bei Tiefe 0."""
        world.get_room("kitchen").occupants.add("cat_01")
        agent.process_inbox(tick=0)  # leere Inbox → nichts gehört
        mock_resp = {"message": {"content": '{"action": "speak", "message": "Hallo?"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.causality_depth == 0

    @pytest.mark.asyncio
    async def test_reactive_speak_increments_depth(self, agent, world):
        """Antwort auf eine gehörte Äußerung bei Tiefe d → eigene Äußerung Tiefe d+1."""
        world.get_room("kitchen").occupants.add("cat_01")
        agent.inbox.append(
            BaseEvent(
                type=EventType.AGENT_SPEAK,
                source="dog_01",
                payload={"message": "Wuff!"},
                causality_depth=2,
            )
        )
        agent.process_inbox(tick=1)
        mock_resp = {"message": {"content": '{"action": "speak", "message": "Miau!"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=1)
        assert event is not None
        assert event.causality_depth == 3

    @pytest.mark.asyncio
    async def test_speak_suppressed_at_max_depth(self, agent, world):
        """Gehörte Tiefe am Limit → Antwort wird unterdrückt, Kette bricht."""
        from nanosim.models import MAX_CAUSALITY_DEPTH

        world.get_room("kitchen").occupants.add("cat_01")
        agent.inbox.append(
            BaseEvent(
                type=EventType.AGENT_SPEAK,
                source="dog_01",
                payload={"message": "Echo"},
                causality_depth=MAX_CAUSALITY_DEPTH,
            )
        )
        agent.process_inbox(tick=2)
        mock_resp = {"message": {"content": '{"action": "speak", "message": "Echo"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=2)
        assert event is None  # unterdrückt → kein weiteres Echo

    @pytest.mark.asyncio
    async def test_reactive_speak_records_caused_by(self, agent, world):
        """Antwort auf eine gehörte Äußerung trägt deren id als caused_by."""
        world.get_room("kitchen").occupants.add("cat_01")
        agent.inbox.append(
            BaseEvent(
                id="evt_dog_hello",
                type=EventType.AGENT_SPEAK,
                source="dog_01",
                payload={"message": "Wuff!"},
                causality_depth=1,
            )
        )
        agent.process_inbox(tick=1)
        mock_resp = {"message": {"content": '{"action": "speak", "message": "Miau!"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=1)
        assert event is not None
        assert event.caused_by == "evt_dog_hello"

    @pytest.mark.asyncio
    async def test_spontaneous_speak_has_no_cause(self, agent, world):
        """Spontane Äußerung (nichts gehört) hat kein caused_by."""
        world.get_room("kitchen").occupants.add("cat_01")
        agent.process_inbox(tick=0)
        mock_resp = {"message": {"content": '{"action": "speak", "message": "Hallo?"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=0)
        assert event is not None
        assert event.caused_by is None

    @pytest.mark.asyncio
    async def test_non_speak_action_not_suppressed_at_max_depth(self, agent, world):
        """move/use/rest bilden keine Echo-Ketten und bleiben auch tief erlaubt."""
        from nanosim.models import MAX_CAUSALITY_DEPTH

        world.get_room("kitchen").occupants.add("cat_01")
        agent.inbox.append(
            BaseEvent(
                type=EventType.AGENT_SPEAK,
                source="dog_01",
                payload={"message": "Echo"},
                causality_depth=MAX_CAUSALITY_DEPTH,
            )
        )
        agent.process_inbox(tick=3)
        mock_resp = {"message": {"content": '{"action": "move", "target": "north"}'}}
        with patch.object(
            agent.router._client, "chat", new_callable=AsyncMock, return_value=mock_resp
        ):
            event = await agent.tick(world, tick=3)
        assert event is not None
        assert event.type == EventType.AGENT_MOVE
