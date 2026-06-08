"""Prompt-Builder: Erzeugt den Kontext-Prompt für einen Agenten."""

from __future__ import annotations

from nanosim.core.world import WorldRegistry
from nanosim.models import AgentProfile


def _nearby_others(agent: AgentProfile, world: WorldRegistry) -> list[str]:
    """Tiere in direkt angrenzenden Räumen wahrnehmen (inkl. Richtung).

    Wie ein Tier, das die anderen im Nachbarraum hört/riecht — genug, um
    gezielt zu ihnen zu gehen. Weiter entfernte Tiere bleiben unbemerkt.
    """
    here = world.get_room(agent.location_id)
    lines: list[str] = []
    for direction, room_id in here.exits.items():
        room = world.get_room(room_id)
        for other in sorted(o for o in room.occupants if o != agent.agent_id):
            lines.append(f"{other} ist nebenan in {room.name} (Richtung {direction})")
    return lines


def build_prompt(agent: AgentProfile, world: WorldRegistry) -> str:
    """Baue den vollständigen Situationsprompt für einen Agenten.

    Enthält: Ort, Objekte, andere Agenten (hier & nebenan), Ausgänge, Stats,
    Inventar, Erinnerungen.
    """
    room = world.get_room(agent.location_id)
    others = sorted(a for a in room.occupants if a != agent.agent_id)
    exits = ", ".join(
        f"{direction} → {room_id}"
        for direction, room_id in room.exits.items()
    )

    memory_str = "; ".join(agent.memory[-3:]) if agent.memory else "keine"
    objects_str = ", ".join(room.objects) if room.objects else "keine"
    others_str = ", ".join(others) if others else "niemand"
    nearby = _nearby_others(agent, world)
    nearby_str = "; ".join(nearby) if nearby else "niemand in der Nähe"
    inventory_str = ", ".join(agent.inventory) if agent.inventory else "leer"

    return (
        f"Du bist {agent.name}. {agent.persona}\n\n"
        f"Ort: {room.name} — {room.description}\n"
        f"Objekte hier: {objects_str}\n"
        f"Andere hier: {others_str}\n"
        f"In der Nähe: {nearby_str}\n"
        f"Ausgänge: {exits if exits else 'keine'}\n\n"
        f"Deine Stats: Energie={agent.stats.stamina:.1f}, "
        f"Stimmung={agent.stats.mood:.1f}, Hunger={agent.stats.hunger:.1f}\n"
        f"Dein Inventar: {inventory_str}\n"
        f"Erinnerungen: {memory_str}\n\n"
        f"Was tust du? Wähle EINE Aktion: speak, move, use, rest, idle.\n"
        f"- speak: message=was du sagst\n"
        f"- move: target=Richtung (z.B. north, south)\n"
        f"- use: target=Objekt-Name\n"
        f"- rest: keine Felder nötig\n"
        f"- idle: keine Felder nötig"
    )


def build_system_prompt(agent: AgentProfile) -> str:
    """Baue den System-Prompt für einen Agenten (inkl. geselligem Charakter)."""
    return (
        f"Du bist {agent.name}. {agent.persona} "
        "Du bist ein geselliges Wesen: Sind andere Tiere bei dir im Raum, "
        "sprich sie an und reagiere auf das, was sie gerade gesagt haben. "
        "Bist du allein, zieht es dich zu den anderen — geh in ihre Richtung, "
        "um sie zu treffen, statt auf der Stelle zu bleiben. "
        "Antworte immer auf Deutsch."
    )
