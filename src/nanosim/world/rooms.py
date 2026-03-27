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
