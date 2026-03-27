"""Integrationstest: Vollständiger Tick-Durchlauf mit echtem Ollama.

Testet das Zusammenspiel aller Komponenten:
- WorldRegistry + Rooms
- AgentProfile + Stats + Memory
- LlamaRouter mit echtem Ollama-Call
- EventBus mit Location-Filterung
- Stats-Decay

Voraussetzung: Ollama läuft auf localhost:11434 mit llama3.1:8b
"""

import asyncio
import logging

import pytest

from nanosim.core.events import EventBus
from nanosim.core.world import WorldRegistry
from nanosim.llm.router import LlamaRouter
from nanosim.models import (
    ActionType,
    AgentAction,
    AgentProfile,
    AgentStats,
    BaseEvent,
    EventType,
    Room,
)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s")
logger = logging.getLogger("test_full_loop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_world() -> WorldRegistry:
    world = WorldRegistry()
    world.add_room(Room(
        room_id="kitchen",
        name="Küche",
        description="Eine warme Küche mit einem Futternapf und einem Fenster.",
        objects=["futternapf", "fenster"],
        exits={"north": "garden"},
    ))
    world.add_room(Room(
        room_id="garden",
        name="Garten",
        description="Ein sonniger Garten mit Gras und einem Schmetterling.",
        objects=["gras", "schmetterling"],
        exits={"south": "kitchen"},
    ))
    return world


def build_agents() -> list[AgentProfile]:
    return [
        AgentProfile(
            agent_id="cat_01",
            name="Whiskers",
            persona="Du bist eine neugierige, verspielte Katze. Du liebst Schmetterlinge und Milch.",
            location_id="kitchen",
            stats=AgentStats(stamina=0.8, mood=0.9, hunger=0.3),
        ),
        AgentProfile(
            agent_id="dog_01",
            name="Bello",
            persona="Du bist ein treuer, etwas tollpatschiger Hund. Du liebst Stöckchen und Fressen.",
            location_id="garden",
            stats=AgentStats(stamina=0.9, mood=0.7, hunger=0.5),
        ),
    ]


def decay_stats(stats: AgentStats) -> AgentStats:
    return stats.model_copy(update={
        "hunger": min(1.0, stats.hunger + 0.05),
        "stamina": max(0.0, stats.stamina - 0.03),
        "mood": max(0.0, stats.mood - 0.02 * stats.hunger),
    })


def build_prompt(agent: AgentProfile, world: WorldRegistry) -> str:
    room = world.get_room(agent.location_id)
    others = [a for a in room.occupants if a != agent.agent_id]
    exits = ", ".join(f"{direction} → {room_id}" for direction, room_id in room.exits.items())

    return (
        f"Du bist {agent.name}. {agent.persona}\n\n"
        f"Ort: {room.name} — {room.description}\n"
        f"Objekte hier: {', '.join(room.objects) if room.objects else 'keine'}\n"
        f"Andere hier: {', '.join(others) if others else 'niemand'}\n"
        f"Ausgänge: {exits if exits else 'keine'}\n\n"
        f"Deine Stats: Energie={agent.stats.stamina:.1f}, Stimmung={agent.stats.mood:.1f}, Hunger={agent.stats.hunger:.1f}\n"
        f"Dein Inventar: {', '.join(agent.inventory) if agent.inventory else 'leer'}\n"
        f"Erinnerungen: {'; '.join(agent.memory[-3:]) if agent.memory else 'keine'}\n\n"
        f"Was tust du? Wähle EINE Aktion: speak, move, use, rest, idle.\n"
        f"- speak: message=was du sagst\n"
        f"- move: target=Richtung (z.B. north, south)\n"
        f"- use: target=Objekt-Name\n"
        f"- rest: keine Felder nötig\n"
        f"- idle: keine Felder nötig"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_full_tick_loop():
    """Simuliere 3 Ticks mit echtem Ollama."""

    router = LlamaRouter(model="llama3.1:8b")
    world = build_world()
    agents = build_agents()
    bus = EventBus()

    # Agenten in Rooms platzieren
    for agent in agents:
        world.get_room(agent.location_id).occupants.add(agent.agent_id)

    # EventBus: jeder Agent sammelt empfangene Events
    agent_inbox: dict[str, list[BaseEvent]] = {a.agent_id: [] for a in agents}
    agent_map: dict[str, AgentProfile] = {a.agent_id: a for a in agents}

    for agent in agents:
        aid = agent.agent_id
        bus.subscribe(
            aid,
            handler=lambda e, _aid=aid: _collect(agent_inbox[_aid], e),
            location_fn=lambda _aid=aid: agent_map[_aid].location_id,
        )

    num_ticks = 3
    all_actions: list[dict] = []

    for tick in range(num_ticks):
        logger.info("=" * 60)
        logger.info("TICK %d", tick)
        logger.info("=" * 60)

        # 1) Stats-Decay
        for agent in agents:
            agent.stats = decay_stats(agent.stats)
            logger.info("[%s] Stats nach Decay: stamina=%.2f mood=%.2f hunger=%.2f",
                        agent.name, agent.stats.stamina, agent.stats.mood, agent.stats.hunger)

        # 2) Inbox verarbeiten → Memory
        for agent in agents:
            for event in agent_inbox[agent.agent_id]:
                memory_line = f"Tick {tick}: {event.source} hat gesagt: {event.payload.get('message', '?')}"
                agent.add_memory(memory_line)
                logger.info("[%s] Memory hinzugefügt: %s", agent.name, memory_line)
            agent_inbox[agent.agent_id].clear()

        # 3) Jeder Agent denkt (sequentiell — Semaphore)
        for agent in agents:
            prompt = build_prompt(agent, world)
            logger.info("[%s] Prompt:\n%s", agent.name, prompt)

            action = await router.think(
                prompt=prompt,
                response_model=AgentAction,
                system=f"Du bist {agent.name}. {agent.persona} Antworte immer auf Deutsch.",
            )

            if action is None:
                logger.warning("[%s] LLM hat kein valides JSON geliefert → idle", agent.name)
                action = AgentAction(action=ActionType.IDLE)

            logger.info("[%s] Action: %s (target=%s, message=%s)",
                        agent.name, action.action.value, action.target, action.message)

            record = {"tick": tick, "agent": agent.name, "action": action.action.value,
                      "target": action.target, "message": action.message}
            all_actions.append(record)

            # 4) Action ausführen
            event = await execute_action(agent, action, world, tick)
            if event:
                await bus.publish(event)
                logger.info("[%s] Event published: %s", agent.name, event.type.value)

        # 5) Events zustellen
        await bus.drain()

    # ---------------------------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("ZUSAMMENFASSUNG: %d Ticks, %d Actions", num_ticks, len(all_actions))
    for rec in all_actions:
        logger.info("  Tick %d | %-10s | %-6s | target=%s | msg=%s",
                     rec["tick"], rec["agent"], rec["action"], rec["target"], rec["message"])

    # Grundlegende Prüfungen
    assert len(all_actions) == num_ticks * len(agents), "Jeder Agent muss pro Tick eine Action haben"

    for rec in all_actions:
        assert rec["action"] in {"speak", "move", "use", "rest", "idle"}, \
            f"Unbekannte Action: {rec['action']}"

    # Stats müssen sich verändert haben
    for agent in agents:
        assert agent.stats.hunger > 0.3, f"{agent.name} Hunger hätte steigen müssen"
        assert agent.stats.stamina < 0.9, f"{agent.name} Stamina hätte sinken müssen"

    logger.info("ALLE CHECKS BESTANDEN")


async def execute_action(
    agent: AgentProfile, action: AgentAction, world: WorldRegistry, tick: int
) -> BaseEvent | None:
    """Führe eine Agent-Action aus und gib das resultierende Event zurück."""

    if action.action == ActionType.SPEAK:
        agent.add_memory(f"Tick {tick}: Ich sagte: '{action.message}'")
        return BaseEvent(
            type=EventType.AGENT_SPEAK,
            source=agent.agent_id,
            location_id=agent.location_id,
            payload={"message": action.message or "..."},
        )

    elif action.action == ActionType.MOVE:
        room = world.get_room(agent.location_id)
        target_dir = action.target or ""
        target_room_id = room.exits.get(target_dir)
        if target_room_id:
            old_loc = agent.location_id
            world.move_agent(agent.agent_id, old_loc, target_room_id)
            agent.location_id = target_room_id
            agent.add_memory(f"Tick {tick}: Bin von {old_loc} nach {target_room_id} gegangen")
            return BaseEvent(
                type=EventType.AGENT_MOVE,
                source=agent.agent_id,
                location_id=old_loc,
                payload={"from": old_loc, "to": target_room_id},
            )
        else:
            agent.add_memory(f"Tick {tick}: Wollte nach {target_dir} gehen, aber kein Ausgang")
            return None

    elif action.action == ActionType.USE:
        agent.add_memory(f"Tick {tick}: Habe {action.target} benutzt")
        return BaseEvent(
            type=EventType.AGENT_USE,
            source=agent.agent_id,
            location_id=agent.location_id,
            payload={"object": action.target or "?"},
        )

    elif action.action == ActionType.REST:
        agent.stats = agent.stats.model_copy(update={
            "stamina": min(1.0, agent.stats.stamina + 0.2),
            "hunger": min(1.0, agent.stats.hunger + 0.05),
        })
        agent.add_memory(f"Tick {tick}: Habe mich ausgeruht")
        return BaseEvent(
            type=EventType.AGENT_REST,
            source=agent.agent_id,
            location_id=agent.location_id,
        )

    # IDLE
    return None


async def _collect(inbox: list[BaseEvent], event: BaseEvent) -> None:
    inbox.append(event)
