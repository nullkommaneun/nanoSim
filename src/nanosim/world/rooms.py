"""Vordefinierte Room-Presets und Terrarium-Layouts."""

from __future__ import annotations

from nanosim.core.world import WorldRegistry
from nanosim.models import Room


# ---------------------------------------------------------------------------
# Einzelne Rooms
# ---------------------------------------------------------------------------

KITCHEN = Room(
    room_id="kitchen",
    name="Küche",
    description="Eine warme Küche mit einem Futternapf und einem Fenster. Es riecht nach Essen.",
    objects=["futternapf", "fenster", "wasserschale"],
    exits={"north": "garden", "east": "living_room"},
)

GARDEN = Room(
    room_id="garden",
    name="Garten",
    description="Ein sonniger Garten mit Gras, Blumen und einem Schmetterling.",
    objects=["gras", "schmetterling", "blume", "stein"],
    exits={"south": "kitchen", "east": "balcony"},
)

LIVING_ROOM = Room(
    room_id="living_room",
    name="Wohnzimmer",
    description="Ein gemütliches Wohnzimmer mit einem Sofa, einem Kratzbaum und einem Teppich.",
    objects=["sofa", "kratzbaum", "teppich", "wollknäuel"],
    exits={"west": "kitchen", "north": "balcony"},
)

BALCONY = Room(
    room_id="balcony",
    name="Balkon",
    description="Ein kleiner Balkon mit Blick auf die Straße. Vögel zwitschern.",
    objects=["blumentopf", "vogelhaus"],
    exits={"south": "living_room", "west": "garden"},
)


# ---------------------------------------------------------------------------
# Layouts (vorgefertigte Welten)
# ---------------------------------------------------------------------------


def create_default_world() -> WorldRegistry:
    """Erstelle das Standard-Terrarium mit 4 Räumen.

    Layout:
        garden ──── balcony
          │            │
        kitchen ── living_room
    """
    world = WorldRegistry()
    world.add_room(KITCHEN.model_copy(deep=True))
    world.add_room(GARDEN.model_copy(deep=True))
    world.add_room(LIVING_ROOM.model_copy(deep=True))
    world.add_room(BALCONY.model_copy(deep=True))
    return world


def create_simple_world() -> WorldRegistry:
    """Erstelle ein minimales Terrarium mit 2 Räumen (zum Testen).

    Layout:
        kitchen ── garden
    """
    kitchen = Room(
        room_id="kitchen",
        name="Küche",
        description="Eine warme Küche mit einem Futternapf.",
        objects=["futternapf", "fenster"],
        exits={"north": "garden"},
    )
    garden = Room(
        room_id="garden",
        name="Garten",
        description="Ein sonniger Garten mit Gras und einem Schmetterling.",
        objects=["gras", "schmetterling"],
        exits={"south": "kitchen"},
    )
    world = WorldRegistry()
    world.add_room(kitchen)
    world.add_room(garden)
    return world


# ---------------------------------------------------------------------------
# Experiment-Topologien (für Welt-Layout-Experimente)
# ---------------------------------------------------------------------------


def create_corridor_world(n: int = 6) -> WorldRegistry:
    """Linearer Korridor aus n Räumen (r0 ── r1 ── … ── r{n-1}).

    Maximaler Durchmesser (n-1) — fordert die Mehrschritt-Navigation am stärksten.
    """
    world = WorldRegistry()
    for i in range(n):
        exits: dict[str, str] = {}
        if i > 0:
            exits["west"] = f"r{i - 1}"
        if i < n - 1:
            exits["east"] = f"r{i + 1}"
        world.add_room(
            Room(
                room_id=f"r{i}",
                name=f"Raum {i}",
                description=f"Ein Raum im Korridor (Position {i}).",
                exits=exits,
            )
        )
    return world


def create_hub_world() -> WorldRegistry:
    """Stern: ein zentraler Treffpunkt (hub) mit 4 Außenräumen.

    Jeder Außenraum erreicht jeden anderen in 2 Schritten über den Hub.
    """
    world = WorldRegistry()
    world.add_room(
        Room(
            room_id="hub",
            name="Treffpunkt",
            description="Ein zentraler Treffpunkt, von dem alle Räume abgehen.",
            exits={"north": "a", "east": "b", "south": "c", "west": "d"},
        )
    )
    # Rückrichtung jeweils entgegengesetzt zur Hub-Richtung.
    back = {"a": "south", "b": "west", "c": "north", "d": "east"}
    for rid, direction in back.items():
        world.add_room(
            Room(
                room_id=rid,
                name=f"Außenraum {rid.upper()}",
                description="Ein Außenraum am Treffpunkt.",
                exits={direction: "hub"},
            )
        )
    return world


def create_twin_cluster_world() -> WorldRegistry:
    """Zwei dichte 3er-Cluster (Dreiecke), verbunden durch eine einzelne Brücke.

    Testet, ob zwei Gruppen über die Engstelle zueinanderfinden.
    """
    world = WorldRegistry()
    # Cluster A (Dreieck a0-a1-a2), a2 hat die Brücke zu b0.
    world.add_room(Room(room_id="a0", name="A0", exits={"east": "a1", "south": "a2"}))
    world.add_room(Room(room_id="a1", name="A1", exits={"west": "a0", "south": "a2"}))
    world.add_room(
        Room(room_id="a2", name="A2", exits={"north": "a0", "up": "a1", "bridge": "b0"})
    )
    # Cluster B (Dreieck b0-b1-b2), b0 hat die Brücke zu a2.
    world.add_room(
        Room(room_id="b0", name="B0", exits={"bridge": "a2", "east": "b1", "south": "b2"})
    )
    world.add_room(Room(room_id="b1", name="B1", exits={"west": "b0", "south": "b2"}))
    world.add_room(Room(room_id="b2", name="B2", exits={"north": "b0", "up": "b1"}))
    return world


# Registry der Layouts: Name -> (Factory, Start-Räume für die 3 Default-Agenten).
# Start-Räume None = die Agenten behalten ihre eigenen Persona-Standorte (default).
WORLD_LAYOUTS: dict[str, tuple] = {
    "default": (create_default_world, None),
    "corridor6": (lambda: create_corridor_world(6), ["r0", "r2", "r5"]),
    "hub5": (create_hub_world, ["a", "b", "c"]),
    "twin": (create_twin_cluster_world, ["a0", "a1", "b2"]),
}
