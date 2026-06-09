"""Tests für die Welt-Layouts (Experiment-Topologien)."""

from nanosim.agents.prompt import _first_step_directions
from nanosim.world.rooms import (
    WORLD_LAYOUTS,
    create_corridor_world,
    create_hub_world,
    create_twin_cluster_world,
)


class TestLayouts:
    def test_corridor_has_n_rooms_and_full_distance(self):
        w = create_corridor_world(6)
        assert len(w.all_rooms()) == 6
        dirs = _first_step_directions(w, "r0")
        assert dirs["r5"][0] == 5  # ganz am anderen Ende, 5 Schritte

    def test_hub_connects_outer_rooms_in_two_steps(self):
        w = create_hub_world()
        assert len(w.all_rooms()) == 5  # hub + 4 außen
        dirs = _first_step_directions(w, "a")
        assert dirs["hub"][0] == 1
        assert dirs["b"][0] == 2  # außen→außen über den Hub

    def test_twin_clusters_are_reachable_across_bridge(self):
        w = create_twin_cluster_world()
        assert len(w.all_rooms()) == 6
        dirs = _first_step_directions(w, "a0")
        assert "b2" in dirs  # andere Gruppe ist erreichbar
        assert dirs["b2"][0] >= 2

    def test_all_layouts_bidirectional(self):
        """In jedem Layout ist jede Verbindung in beide Richtungen begehbar."""
        for factory, _ in WORLD_LAYOUTS.values():
            w = factory()
            ids = {r.room_id for r in w.all_rooms()}
            for room in w.all_rooms():
                for target in room.exits.values():
                    assert target in ids  # Ausgang zeigt auf existierenden Raum
                    back = w.get_room(target)
                    assert room.room_id in back.exits.values()  # Rückweg existiert

    def test_registry_has_expected_layouts(self):
        assert {"default", "corridor6", "hub5", "twin"} <= set(WORLD_LAYOUTS)


class TestBuildWorldWithLayout:
    def test_build_corridor_places_agents(self):
        from nanosim.main import _build_world_and_profiles

        world, profiles, tick = _build_world_and_profiles(None, world_name="corridor6")
        assert tick == 0
        assert len(world.all_rooms()) == 6
        # jeder Agent steht in einem existierenden Raum des Layouts
        for p in profiles:
            assert p.agent_id in world.agents_in_room(p.location_id)

    def test_default_world_unchanged(self):
        from nanosim.main import _build_world_and_profiles

        world, profiles, _ = _build_world_and_profiles(None)
        assert world.get_room("kitchen").name == "Küche"
        assert "cat_01" in world.agents_in_room("kitchen")
