"""Speichern und Laden des Weltzustands als JSON.

Ein Snapshot hält alles, was nötig ist, um eine Simulation nach einem
Neustart fortzusetzen: den Tick-Zähler, alle Rooms und alle Agent-Profile
(inkl. Stats, Gedächtnis und Position).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from nanosim.core.world import WorldRegistry
from nanosim.models import AgentProfile, Room


class Snapshot(BaseModel):
    """Vollständiger, serialisierbarer Zustand einer Simulation."""

    tick_count: int = 0
    rooms: list[Room] = Field(default_factory=list)
    profiles: list[AgentProfile] = Field(default_factory=list)


def save_snapshot(
    path: str | Path,
    *,
    world: WorldRegistry,
    profiles: list[AgentProfile],
    tick_count: int,
) -> None:
    """Schreibe den aktuellen Weltzustand als JSON auf die Platte."""
    snapshot = Snapshot(
        tick_count=tick_count,
        rooms=world.all_rooms(),
        profiles=profiles,
    )
    Path(path).write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")


def load_snapshot(path: str | Path) -> Snapshot:
    """Lade einen Snapshot aus einer JSON-Datei."""
    text = Path(path).read_text(encoding="utf-8")
    return Snapshot.model_validate_json(text)


def world_from_snapshot(snapshot: Snapshot) -> WorldRegistry:
    """Baue eine WorldRegistry aus einem Snapshot wieder auf."""
    world = WorldRegistry()
    for room in snapshot.rooms:
        world.add_room(room)
    return world
