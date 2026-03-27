"""Tests für die WorldRegistry."""

import pytest

from nanosim.core.world import WorldRegistry
from nanosim.models import Room


@pytest.fixture
def world():
    w = WorldRegistry()
    w.add_room(Room(
        room_id="kitchen", name="Küche",
        exits={"north": "garden"},
    ))
    w.add_room(Room(
        room_id="garden", name="Garten",
        exits={"south": "kitchen"},
    ))
    return w


class TestWorldRegistry:
    def test_get_room(self, world: WorldRegistry):
        room = world.get_room("kitchen")
        assert room.name == "Küche"

    def test_get_room_not_found(self, world: WorldRegistry):
        with pytest.raises(KeyError):
            world.get_room("nonexistent")

    def test_all_rooms(self, world: WorldRegistry):
        rooms = world.all_rooms()
        assert len(rooms) == 2

    def test_move_agent(self, world: WorldRegistry):
        # Agent in kitchen platzieren
        world.get_room("kitchen").occupants.add("cat_01")
        assert "cat_01" in world.agents_in_room("kitchen")

        # Agent nach garden bewegen
        world.move_agent("cat_01", "kitchen", "garden")
        assert "cat_01" not in world.agents_in_room("kitchen")
        assert "cat_01" in world.agents_in_room("garden")

    def test_agents_in_room_empty(self, world: WorldRegistry):
        assert world.agents_in_room("kitchen") == set()

    def test_move_agent_idempotent_discard(self, world: WorldRegistry):
        """move_agent soll nicht crashen wenn Agent nicht im from_room ist."""
        world.move_agent("ghost", "kitchen", "garden")
        assert "ghost" in world.agents_in_room("garden")
