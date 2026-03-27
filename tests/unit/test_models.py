"""Tests für die zentralen Pydantic-Modelle."""

import pytest
from pydantic import ValidationError

from nanosim.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    AgentStats,
    BaseEvent,
    EventType,
    Room,
)


# ---------------------------------------------------------------------------
# BaseEvent
# ---------------------------------------------------------------------------

class TestBaseEvent:
    def test_create_minimal(self):
        e = BaseEvent(type=EventType.TICK, source="system")
        assert e.source == "system"
        assert e.target is None
        assert e.location_id is None
        assert e.payload == {}
        assert e.causality_depth == 0
        assert len(e.id) == 12

    def test_create_with_location(self):
        e = BaseEvent(
            type=EventType.AGENT_SPEAK,
            source="cat_01",
            location_id="kitchen",
            payload={"message": "Miau!"},
        )
        assert e.location_id == "kitchen"
        assert e.payload["message"] == "Miau!"

    def test_unique_ids(self):
        e1 = BaseEvent(type=EventType.TICK, source="system")
        e2 = BaseEvent(type=EventType.TICK, source="system")
        assert e1.id != e2.id


# ---------------------------------------------------------------------------
# AgentStats
# ---------------------------------------------------------------------------

class TestAgentStats:
    def test_defaults(self):
        stats = AgentStats()
        assert stats.stamina == 1.0
        assert stats.mood == 1.0
        assert stats.hunger == 0.0

    def test_clamp_upper(self):
        with pytest.raises(ValidationError):
            AgentStats(stamina=1.5)

    def test_clamp_lower(self):
        with pytest.raises(ValidationError):
            AgentStats(mood=-0.1)

    def test_valid_custom(self):
        stats = AgentStats(stamina=0.5, mood=0.3, hunger=0.8)
        assert stats.stamina == 0.5


# ---------------------------------------------------------------------------
# AgentAction
# ---------------------------------------------------------------------------

class TestAgentAction:
    def test_speak(self):
        a = AgentAction(action=ActionType.SPEAK, message="Hallo!")
        assert a.action == ActionType.SPEAK
        assert a.message == "Hallo!"

    def test_move(self):
        a = AgentAction(action=ActionType.MOVE, target="garden")
        assert a.target == "garden"

    def test_idle_minimal(self):
        a = AgentAction(action=ActionType.IDLE)
        assert a.target is None
        assert a.message is None


# ---------------------------------------------------------------------------
# AgentProfile
# ---------------------------------------------------------------------------

class TestAgentProfile:
    def test_create(self):
        agent = AgentProfile(
            agent_id="cat_01",
            name="Whiskers",
            persona="Du bist eine neugierige Katze.",
            location_id="kitchen",
        )
        assert agent.agent_id == "cat_01"
        assert agent.stats.stamina == 1.0
        assert agent.inventory == []
        assert agent.memory == []

    def test_add_memory_basic(self):
        agent = AgentProfile(
            agent_id="cat_01", name="Whiskers",
            persona="Katze", location_id="kitchen",
        )
        agent.add_memory("Tick 1: Habe Milch getrunken")
        assert len(agent.memory) == 1
        assert "Milch" in agent.memory[0]

    def test_add_memory_fifo(self):
        agent = AgentProfile(
            agent_id="cat_01", name="Whiskers",
            persona="Katze", location_id="kitchen",
        )
        for i in range(15):
            agent.add_memory(f"Tick {i}: Event {i}")
        assert len(agent.memory) == 10
        # Älteste Einträge (0-4) sollten weg sein
        assert "Event 0" not in agent.memory[0]
        assert "Event 5" in agent.memory[0]
        assert "Event 14" in agent.memory[-1]


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------

class TestRoom:
    def test_create(self):
        room = Room(room_id="kitchen", name="Küche", description="Eine gemütliche Küche")
        assert room.room_id == "kitchen"
        assert room.occupants == set()
        assert room.objects == []
        assert room.exits == {}

    def test_with_exits_and_objects(self):
        room = Room(
            room_id="garden",
            name="Garten",
            objects=["ball", "blume"],
            exits={"south": "kitchen"},
        )
        assert len(room.objects) == 2
        assert room.exits["south"] == "kitchen"

    def test_occupants_set_behavior(self):
        room = Room(room_id="kitchen", name="Küche")
        room.occupants.add("cat_01")
        room.occupants.add("cat_01")  # Duplikat
        assert len(room.occupants) == 1
