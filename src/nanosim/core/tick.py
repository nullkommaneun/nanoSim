"""Tick-Engine: Treibt die Weltzeit voran."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from nanosim.models import ActionType, AgentStats, BaseEvent
from nanosim.trace import AgentSnapshot, Decision, TickRecord

if TYPE_CHECKING:
    from nanosim.agents.base import BaseAgent
    from nanosim.core.events import EventBus
    from nanosim.core.world import WorldRegistry
    from nanosim.trace import TraceWriter

logger = logging.getLogger(__name__)


def decay_stats(stats: AgentStats) -> AgentStats:
    """Berechne den natürlichen Verfall der Agent-Stats pro Tick.

    - Hunger steigt langsam (+0.05)
    - Stamina sinkt langsam (-0.03)
    - Mood sinkt proportional zum Hunger
    """
    return stats.model_copy(update={
        "hunger": min(1.0, stats.hunger + 0.05),
        "stamina": max(0.0, stats.stamina - 0.03),
        "mood": max(0.0, stats.mood - 0.02 * stats.hunger),
    })


class TickEngine:
    """Asynchroner Loop der die Weltzeit vorantreibt.

    Pro Tick:
    1. Stats-Decay für alle Agenten
    2. Inbox verarbeiten → Agent-Memory
    3. Jeder Agent denkt und handelt (sequentiell, Reihenfolge zufällig)
    4. Events über den Bus verteilen
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        world: WorldRegistry,
        bus: EventBus,
        recorder: TraceWriter | None = None,
    ) -> None:
        self.agents = agents
        self.world = world
        self.bus = bus
        self.recorder = recorder
        self.tick_count: int = 0

    async def run(self, num_ticks: int | None = None) -> None:
        """Laufe num_ticks Ticks, oder endlos wenn None."""
        tick = 0
        while num_ticks is None or tick < num_ticks:
            await self.step()
            tick += 1

    async def step(self) -> None:
        """Führe einen einzelnen Tick aus."""
        logger.info("=" * 50)
        logger.info("TICK %d", self.tick_count)
        logger.info("=" * 50)

        # 1) Stats-Decay
        for agent in self.agents:
            agent.profile.stats = decay_stats(agent.profile.stats)
            s = agent.profile.stats
            logger.info(
                "[%s] Stats: stamina=%.2f mood=%.2f hunger=%.2f",
                agent.profile.name, s.stamina, s.mood, s.hunger,
            )

        # 2) Inbox → Memory
        for agent in self.agents:
            agent.process_inbox(self.tick_count)

        # 3) Agenten denken und handeln (zufällige Reihenfolge für Fairness)
        shuffled = list(self.agents)
        random.shuffle(shuffled)

        emitted: list[BaseEvent] = []
        for agent in shuffled:
            event = await agent.tick(self.world, self.tick_count)
            if event is not None:
                emitted.append(event)
                await self.bus.publish(event)
                logger.info(
                    "[%s] → %s", agent.profile.name, event.type.value,
                )

        # 4) Events zustellen
        await self.bus.drain()

        # 5) Optional: diesen Tick aufzeichnen
        if self.recorder is not None:
            self._record_tick(emitted)

        self.tick_count += 1

    def _record_tick(self, emitted: list[BaseEvent]) -> None:
        """Den aktuellen Tick als TickRecord an den Recorder geben."""
        snapshots = [
            AgentSnapshot(
                agent_id=a.profile.agent_id,
                name=a.profile.name,
                location_id=a.profile.location_id,
                stats=a.profile.stats,
            )
            for a in self.agents
        ]
        decisions = [
            Decision(
                agent_id=a.profile.agent_id,
                action=a.last_action.action if a.last_action else ActionType.IDLE,
                target=a.last_action.target if a.last_action else None,
                message=a.last_action.message if a.last_action else None,
            )
            for a in self.agents
        ]
        record = TickRecord(
            tick=self.tick_count,
            agents=snapshots,
            decisions=decisions,
            events=emitted,
        )
        self.recorder.record_tick(record)
