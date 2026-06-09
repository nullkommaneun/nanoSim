"""Objekt-Effekte: Was passiert, wenn ein Agent ein Objekt benutzt.

Jedes Objekt verändert die Stats des Agenten um feste Deltas (geklemmt auf
[0, 1]). So bekommt die Welt eine deterministische, beobachtbare Spiel-Dynamik
— unabhängig von der Qualität des LLM. Objekte ohne Eintrag haben keine
Wirkung (z. B. reine Deko wie "fenster" oder "stein").
"""

from __future__ import annotations

from nanosim.models import AgentStats

# Wirkung pro Objekt: Stat-Name → Delta (positiv hebt, negativ senkt).
OBJECT_EFFECTS: dict[str, dict[str, float]] = {
    # Küche
    "futternapf": {"hunger": -0.5, "mood": 0.05},  # fressen
    "wasserschale": {"hunger": -0.1},  # trinken (leichte Sättigung)
    # Garten
    "gras": {"hunger": -0.15},  # knabbern
    "schmetterling": {"mood": 0.2, "stamina": -0.05},  # jagen macht Spaß, kostet Kraft
    "blume": {"mood": 0.1},  # schnuppern
    # Wohnzimmer
    "sofa": {"stamina": 0.3, "mood": 0.1},  # ausruhen
    "kratzbaum": {"mood": 0.2, "stamina": 0.05},  # kratzen/klettern
    "teppich": {"stamina": 0.1},  # dösen
    "wollknäuel": {"mood": 0.25, "stamina": -0.05},  # spielen
    # Balkon
    "vogelhaus": {"mood": 0.15},  # Vögel beobachten
}


def apply_object_effect(stats: AgentStats, obj: str) -> AgentStats:
    """Die Wirkung eines Objekts auf die Stats anwenden (geklemmt, ohne Mutation).

    Gibt unveränderte Stats zurück, wenn das Objekt keinen Eintrag hat.
    """
    effect = OBJECT_EFFECTS.get(obj)
    if not effect:
        return stats

    updates: dict[str, float] = {}
    for stat, delta in effect.items():
        new_val = getattr(stats, stat) + delta
        updates[stat] = max(0.0, min(1.0, new_val))
    return stats.model_copy(update=updates)
