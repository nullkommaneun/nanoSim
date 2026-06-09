"""Prompt-Builder: Erzeugt den Kontext-Prompt für einen Agenten."""

from __future__ import annotations

from collections import deque

from nanosim.core.world import WorldRegistry
from nanosim.models import AgentProfile


def _first_step_directions(
    world: WorldRegistry, start_id: str,
) -> dict[str, tuple[int, str | None]]:
    """Breitensuche über die Räume: für jeden erreichbaren Raum die Entfernung
    (in Räumen) und die Richtung des ersten Schritts dorthin.

    Das ist der „soziale Kompass": Auch wer mehrere Räume entfernt ist, kann so
    Schritt für Schritt angesteuert werden.
    """
    visited: dict[str, tuple[int, str | None]] = {start_id: (0, None)}
    queue: deque[str] = deque([start_id])
    while queue:
        rid = queue.popleft()
        dist, first_dir = visited[rid]
        for direction, nxt in world.get_room(rid).exits.items():
            if nxt in visited:
                continue
            # Im ersten Schritt ist die Richtung der Schritt selbst, danach
            # erben wir die Richtung des allerersten Schritts.
            visited[nxt] = (dist + 1, direction if dist == 0 else first_dir)
            queue.append(nxt)
    return visited


def _sensed_others(agent: AgentProfile, world: WorldRegistry) -> list[str]:
    """Alle erreichbaren Tiere wahrnehmen — mit Entfernung und Erst-Schritt-Richtung."""
    dirs = _first_step_directions(world, agent.location_id)
    lines: list[str] = []
    for room in world.all_rooms():
        if room.room_id == agent.location_id or room.room_id not in dirs:
            continue
        dist, direction = dirs[room.room_id]
        for other in sorted(o for o in room.occupants if o != agent.agent_id):
            if dist == 1:
                lines.append(f"{other} ist nebenan in {room.name} (Richtung {direction})")
            else:
                lines.append(
                    f"{other} ist in {room.name}, {dist} Räume entfernt "
                    f"(geh Richtung {direction})"
                )
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
    sensed = _sensed_others(agent, world)
    sensed_str = "; ".join(sensed) if sensed else "niemand erreichbar"
    inventory_str = ", ".join(agent.inventory) if agent.inventory else "leer"

    # Zwei situative Anstöße, die sich gegenseitig ausschließen:
    # - jemand IST da  → bleiben und reden (sonst brechen Gespräche ab)
    # - allein, aber jemand erreichbar → dem Kompass folgen (auch über mehrere Räume)
    if others:
        stay_urge = (
            "\n>> Bleib hier und sprich mit den anderen im Raum — geh jetzt NICHT weg!\n"
        )
    elif sensed:
        stay_urge = (
            f"\n>> Du bist allein — geh zu den anderen Tieren! Folge dem Kompass: {sensed_str}\n"
        )
    else:
        stay_urge = ""

    return (
        f"Du bist {agent.name}. {agent.persona}\n\n"
        f"Ort: {room.name} — {room.description}\n"
        f"Objekte hier: {objects_str}\n"
        f"Andere hier: {others_str}\n"
        f"Andere Tiere (Kompass): {sensed_str}\n"
        f"Ausgänge: {exits if exits else 'keine'}\n"
        f"{stay_urge}\n"
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
        "Du bist ein geselliges Wesen. Sind andere Tiere bei dir im Raum, "
        "bleib bei ihnen und führe ein Gespräch — sprich sie an, antworte auf "
        "das, was sie gerade gesagt haben, und geh nicht weg, solange ihr redet. "
        "Nur wenn du ganz allein bist, zieht es dich zu den anderen: dann geh "
        "in ihre Richtung, um sie zu treffen. "
        "Antworte immer auf Deutsch."
    )
