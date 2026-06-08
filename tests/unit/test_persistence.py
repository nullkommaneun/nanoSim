"""Tests für das Speichern/Laden des Weltzustands (Persistence)."""

import pytest

from nanosim.core.world import WorldRegistry
from nanosim.models import AgentProfile, AgentStats, Room
from nanosim.persistence import (
    Snapshot,
    load_snapshot,
    save_snapshot,
    world_from_snapshot,
)


@pytest.fixture
def world():
    w = WorldRegistry()
    w.add_room(Room(
        room_id="kitchen", name="Küche", description="Eine warme Küche.",
        objects=["futternapf"], exits={"north": "garden"},
    ))
    w.add_room(Room(
        room_id="garden", name="Garten", description="Ein sonniger Garten.",
        objects=["gras"], exits={"south": "kitchen"},
    ))
    w.get_room("kitchen").occupants.add("cat_01")
    return w


@pytest.fixture
def profiles():
    cat = AgentProfile(
        agent_id="cat_01", name="Whiskers", persona="Neugierige Katze.",
        location_id="kitchen", stats=AgentStats(stamina=0.7, mood=0.6, hunger=0.4),
    )
    cat.add_memory("Tick 3: Habe Milch getrunken")
    return [cat]


class TestRoundTrip:
    def test_tick_count_preserved(self, tmp_path, world, profiles):
        path = tmp_path / "save.json"
        save_snapshot(path, world=world, profiles=profiles, tick_count=42)
        snap = load_snapshot(path)
        assert snap.tick_count == 42

    def test_rooms_preserved(self, tmp_path, world, profiles):
        path = tmp_path / "save.json"
        save_snapshot(path, world=world, profiles=profiles, tick_count=0)
        snap = load_snapshot(path)
        room_ids = {r.room_id for r in snap.rooms}
        assert room_ids == {"kitchen", "garden"}

    def test_occupants_stay_a_set(self, tmp_path, world, profiles):
        """occupants ist ein set[str] — muss nach dem Laden wieder ein set sein."""
        path = tmp_path / "save.json"
        save_snapshot(path, world=world, profiles=profiles, tick_count=0)
        snap = load_snapshot(path)
        kitchen = next(r for r in snap.rooms if r.room_id == "kitchen")
        assert kitchen.occupants == {"cat_01"}
        assert isinstance(kitchen.occupants, set)

    def test_profile_stats_and_memory_preserved(self, tmp_path, world, profiles):
        path = tmp_path / "save.json"
        save_snapshot(path, world=world, profiles=profiles, tick_count=0)
        snap = load_snapshot(path)
        cat = next(p for p in snap.profiles if p.agent_id == "cat_01")
        assert cat.stats.hunger == 0.4
        assert cat.name == "Whiskers"
        assert any("Milch" in m for m in cat.memory)

    def test_world_from_snapshot_rebuilds_registry(self, tmp_path, world, profiles):
        path = tmp_path / "save.json"
        save_snapshot(path, world=world, profiles=profiles, tick_count=0)
        snap = load_snapshot(path)
        rebuilt = world_from_snapshot(snap)
        assert isinstance(rebuilt, WorldRegistry)
        assert rebuilt.get_room("kitchen").name == "Küche"
        assert "cat_01" in rebuilt.agents_in_room("kitchen")


class TestResumeHelper:
    """_build_world_and_profiles: frisch bauen oder aus Snapshot fortsetzen."""

    def test_fresh_when_no_load_path(self):
        from nanosim.main import _build_world_and_profiles

        world, profiles, tick = _build_world_and_profiles(None)
        assert tick == 0
        assert len(profiles) > 0
        # Jeder Agent steht in seinem Room als occupant
        for p in profiles:
            assert p.agent_id in world.agents_in_room(p.location_id)

    def test_resume_restores_tick_and_profiles(self, tmp_path, world, profiles):
        from nanosim.main import _build_world_and_profiles

        path = tmp_path / "save.json"
        save_snapshot(path, world=world, profiles=profiles, tick_count=10)

        rebuilt_world, rebuilt_profiles, tick = _build_world_and_profiles(str(path))
        assert tick == 10
        assert {p.agent_id for p in rebuilt_profiles} == {"cat_01"}
        assert "cat_01" in rebuilt_world.agents_in_room("kitchen")


class TestSnapshotModel:
    def test_snapshot_is_json_serializable(self, world, profiles):
        snap = Snapshot(
            tick_count=1, rooms=world.all_rooms(), profiles=profiles,
        )
        text = snap.model_dump_json()
        assert "kitchen" in text
        assert "Whiskers" in text
