"""Tests für World-Presets und Personas."""

from nanosim.world.personas import (
    create_cat,
    create_default_agents,
    create_dog,
    create_parrot,
)
from nanosim.world.rooms import create_default_world, create_simple_world


class TestRoomPresets:
    def test_default_world_has_4_rooms(self):
        world = create_default_world()
        assert len(world.all_rooms()) == 4

    def test_default_world_rooms_connected(self):
        world = create_default_world()
        kitchen = world.get_room("kitchen")
        garden = world.get_room("garden")
        assert kitchen.exits["north"] == "garden"
        assert garden.exits["south"] == "kitchen"

    def test_default_world_all_exits_valid(self):
        """Jeder Exit zeigt auf einen existierenden Room."""
        world = create_default_world()
        room_ids = {r.room_id for r in world.all_rooms()}
        for room in world.all_rooms():
            for direction, target in room.exits.items():
                assert target in room_ids, (
                    f"Room '{room.room_id}' Exit '{direction}' → '{target}' existiert nicht"
                )

    def test_simple_world_has_2_rooms(self):
        world = create_simple_world()
        assert len(world.all_rooms()) == 2

    def test_default_world_deep_copy(self):
        """Zwei Welten teilen sich keine mutable Objekte."""
        w1 = create_default_world()
        w2 = create_default_world()
        w1.get_room("kitchen").occupants.add("test")
        assert "test" not in w2.get_room("kitchen").occupants


class TestPersonas:
    def test_create_cat(self):
        cat = create_cat()
        assert cat.agent_id == "cat_01"
        assert "Katze" in cat.persona

    def test_create_dog(self):
        dog = create_dog()
        assert dog.agent_id == "dog_01"
        assert "Hund" in dog.persona

    def test_create_parrot(self):
        parrot = create_parrot()
        assert parrot.agent_id == "parrot_01"
        assert "Papagei" in parrot.persona

    def test_create_cat_custom_id(self):
        cat = create_cat(agent_id="kitty", name="Luna")
        assert cat.agent_id == "kitty"
        assert cat.name == "Luna"

    def test_default_agents_has_3(self):
        agents = create_default_agents()
        assert len(agents) == 3

    def test_default_agents_unique_ids(self):
        agents = create_default_agents()
        ids = [a.agent_id for a in agents]
        assert len(set(ids)) == 3
