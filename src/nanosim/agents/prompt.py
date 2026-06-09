"""Prompt-Builder: Erzeugt den Kontext-Prompt für einen Agenten."""

from __future__ import annotations

from collections import deque

from nanosim.core.world import WorldRegistry
from nanosim.models import AgentProfile


def _first_step_directions(
    world: WorldRegistry,
    start_id: str,
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
                lines.append(
                    f"{other} ist nebenan in {room.name} (Richtung {direction})"
                )
            else:
                lines.append(
                    f"{other} ist in {room.name}, {dist} Räume entfernt "
                    f"(geh Richtung {direction})"
                )
    return lines


# Sozial-Trieb-Varianten (für Experimente). Jede Variante definiert den
# Charakter-Satz im System-Prompt sowie die situativen Anstöße (bleiben/gehen).
# "baseline" ist verhaltensgleich zum Stand vor der Parametrisierung.
PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {
        "system": (
            "Du bist ein geselliges Wesen. Sind andere Tiere bei dir im Raum, "
            "bleib bei ihnen und führe ein Gespräch — sprich sie an, antworte auf "
            "das, was sie gerade gesagt haben, und geh nicht weg, solange ihr redet. "
            "Nur wenn du ganz allein bist, zieht es dich zu den anderen: dann geh "
            "in ihre Richtung, um sie zu treffen."
        ),
        "stay": "\n>> Bleib hier und sprich mit den anderen im Raum — geh jetzt NICHT weg!\n",
        "move": "\n>> Du bist allein — geh zu den anderen Tieren! Folge dem Kompass: {compass}\n",
    },
    "soft": {
        "system": (
            "Du bist neugierig. Andere Tiere sind manchmal in der Nähe; du kannst "
            "sie ansprechen oder deiner Wege gehen, ganz wie dir ist."
        ),
        "stay": "\n(Andere sind hier im Raum.)\n",
        "move": "\n(Andere Tiere sind in der Nähe: {compass})\n",
    },
    "hard": {
        "system": (
            "Du bist ausgesprochen gesellig. Ist jemand bei dir im Raum, MUSST du "
            "bleiben und mit einer RÜCKFRAGE auf das zuletzt Gesagte antworten — "
            "geh auf keinen Fall weg, solange jemand da ist. Bist du allein, geh "
            "sofort zu den anderen."
        ),
        "stay": "\n>> BLEIB HIER! Stelle den anderen eine Frage zu dem, was sie gerade gesagt haben. Geh NICHT weg!\n",
        "move": "\n>> Du bist allein — geh SOFORT zu den anderen! Kompass: {compass}\n",
    },
    "naming": {
        "system": (
            "Du bist gesellig und höflich. Sind andere bei dir, bleib und sprich "
            "sie mit NAMEN an; antworte direkt auf das, was die zuletzt sprechende "
            "Person gesagt hat. Bist du allein, geh zu den anderen."
        ),
        "stay": "\n>> Bleib hier. Sprich die anderen mit NAMEN an und antworte direkt auf das zuletzt Gesagte!\n",
        "move": "\n>> Du bist allein — geh zu den anderen Tieren (Kompass: {compass})\n",
    },
}


def build_prompt(
    agent: AgentProfile,
    world: WorldRegistry,
    memory_window: int = 3,
    variant: str = "baseline",
) -> str:
    """Baue den vollständigen Situationsprompt für einen Agenten.

    Enthält: Ort, Objekte, andere Agenten (hier & nebenan), Ausgänge, Stats,
    Inventar, Erinnerungen. `memory_window` steuert, wie viele Erinnerungen
    sichtbar sind; `variant` wählt den Sozial-Trieb-Wortlaut.
    """
    room = world.get_room(agent.location_id)
    others = sorted(a for a in room.occupants if a != agent.agent_id)
    exits = ", ".join(
        f"{direction} → {room_id}" for direction, room_id in room.exits.items()
    )

    shown_memory = agent.memory[-memory_window:] if memory_window > 0 else []
    memory_str = "; ".join(shown_memory) if shown_memory else "keine"
    objects_str = ", ".join(room.objects) if room.objects else "keine"
    others_str = ", ".join(others) if others else "niemand"
    sensed = _sensed_others(agent, world)
    sensed_str = "; ".join(sensed) if sensed else "niemand erreichbar"
    inventory_str = ", ".join(agent.inventory) if agent.inventory else "leer"

    texts = PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["baseline"])
    # Zwei situative Anstöße, die sich gegenseitig ausschließen:
    # - jemand IST da  → bleiben und reden (sonst brechen Gespräche ab)
    # - allein, aber jemand erreichbar → dem Kompass folgen (auch über mehrere Räume)
    if others:
        stay_urge = texts["stay"]
    elif sensed:
        stay_urge = texts["move"].format(compass=sensed_str)
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


def build_system_prompt(agent: AgentProfile, variant: str = "baseline") -> str:
    """Baue den System-Prompt für einen Agenten (inkl. geselligem Charakter)."""
    texts = PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["baseline"])
    return f"Du bist {agent.name}. {agent.persona} {texts['system']} Antworte immer auf Deutsch."
