"""BaseAgent: Verbindet Profil, LLM-Router und Welt-Interaktion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nanosim.agents.prompt import build_prompt, build_system_prompt
from nanosim.models import (
    MAX_CAUSALITY_DEPTH,
    ActionType,
    AgentAction,
    AgentProfile,
    BaseEvent,
    EventType,
)
from nanosim.world.objects import apply_object_effect

if TYPE_CHECKING:
    from nanosim.core.world import WorldRegistry
    from nanosim.llm.router import LlamaRouter

logger = logging.getLogger(__name__)


class BaseAgent:
    """Ein Agent im Terrarium.

    Verwaltet sein Profil, empfängt Events über die Inbox,
    und entscheidet via LLM welche Action er ausführt.
    """

    def __init__(self, profile: AgentProfile, router: LlamaRouter) -> None:
        self.profile = profile
        self.router = router
        self.inbox: list[BaseEvent] = []
        # Tiefe der zuletzt gehörten Äußerung (für Causality-Kappung).
        # None = in diesem Tick nichts gehört → frische Kette (Tiefe 0).
        self._heard_speak_depth: int | None = None
        # id der gehörten Äußerung, auf die reagiert wird (für Causality-Baum).
        self._heard_speak_id: str | None = None
        # Zuletzt getroffene Entscheidung (für Trace-Aufzeichnung / Replay).
        self.last_action: AgentAction | None = None

    @property
    def agent_id(self) -> str:
        return self.profile.agent_id

    # ------------------------------------------------------------------
    # Event-Handling
    # ------------------------------------------------------------------

    async def receive_event(self, event: BaseEvent) -> None:
        """Event vom Bus empfangen und in die Inbox legen."""
        self.inbox.append(event)

    def process_inbox(self, tick: int) -> None:
        """Inbox verarbeiten und relevante Infos ins Memory schreiben."""
        # Tiefste gehörte Äußerung dieses Ticks merken (id + Tiefe); None wenn keine.
        heard = [e for e in self.inbox if e.type == EventType.AGENT_SPEAK]
        if heard:
            trigger = max(heard, key=lambda e: e.causality_depth)
            self._heard_speak_depth = trigger.causality_depth
            self._heard_speak_id = trigger.id
        else:
            self._heard_speak_depth = None
            self._heard_speak_id = None

        for event in self.inbox:
            if event.type == EventType.AGENT_SPEAK:
                msg = event.payload.get("message", "?")
                self.profile.add_memory(f"Tick {tick}: {event.source} sagte: '{msg}'")
            elif event.type == EventType.AGENT_MOVE:
                to = event.payload.get("to", "?")
                self.profile.add_memory(f"Tick {tick}: {event.source} ging nach {to}")
            elif event.type == EventType.AGENT_USE:
                obj = event.payload.get("object", "?")
                self.profile.add_memory(f"Tick {tick}: {event.source} benutzte {obj}")
            elif event.type == EventType.AGENT_REST:
                self.profile.add_memory(f"Tick {tick}: {event.source} ruhte sich aus")
        self.inbox.clear()

    # ------------------------------------------------------------------
    # Denken & Handeln
    # ------------------------------------------------------------------

    async def tick(self, world: WorldRegistry, tick: int) -> BaseEvent | None:
        """Einen kompletten Agent-Tick ausführen: denken → handeln → Event."""
        prompt = build_prompt(self.profile, world)
        system = build_system_prompt(self.profile)

        logger.debug("[%s] Denkt nach...", self.profile.name)

        action = await self.router.think(
            prompt=prompt,
            response_model=AgentAction,
            system=system,
        )

        if action is None:
            logger.warning("[%s] Kein valides JSON → idle", self.profile.name)
            action = AgentAction(action=ActionType.IDLE)

        # Entscheidung festhalten (Trace/Replay) — die tatsächlich gewählte Action.
        self.last_action = action

        logger.info(
            "[%s] %s (target=%s, message=%s)",
            self.profile.name, action.action.value, action.target, action.message,
        )

        return self._execute(action, world, tick)

    def _execute(
        self, action: AgentAction, world: WorldRegistry, tick: int,
    ) -> BaseEvent | None:
        """Eine Action ausführen und das resultierende Event zurückgeben."""

        if action.action == ActionType.SPEAK:
            # Antwort auf Gehörtes → eine Ebene tiefer; spontan → Tiefe 0.
            depth = self._heard_speak_depth + 1 if self._heard_speak_depth is not None else 0
            if depth > MAX_CAUSALITY_DEPTH:
                # Kette zu tief → schweigen, damit kein Endlos-Echo entsteht.
                logger.info(
                    "[%s] Schweigt: Reaktions-Kette zu tief (%d)",
                    self.profile.name, depth,
                )
                self.profile.add_memory(
                    f"Tick {tick}: Wollte antworten, hielt aber inne (zu viel Gerede)"
                )
                return None
            self.profile.add_memory(f"Tick {tick}: Ich sagte: '{action.message}'")
            return BaseEvent(
                type=EventType.AGENT_SPEAK,
                source=self.agent_id,
                location_id=self.profile.location_id,
                payload={"message": action.message or "..."},
                causality_depth=depth,
                caused_by=self._heard_speak_id if depth > 0 else None,
            )

        if action.action == ActionType.MOVE:
            return self._execute_move(action, world, tick)

        if action.action == ActionType.USE:
            return self._execute_use(action, world, tick)

        if action.action == ActionType.REST:
            self.profile.stats = self.profile.stats.model_copy(update={
                "stamina": min(1.0, self.profile.stats.stamina + 0.2),
                "hunger": min(1.0, self.profile.stats.hunger + 0.05),
            })
            self.profile.add_memory(f"Tick {tick}: Ruhte mich aus")
            return BaseEvent(
                type=EventType.AGENT_REST,
                source=self.agent_id,
                location_id=self.profile.location_id,
            )

        # IDLE — kein Event
        return None

    def _execute_use(
        self, action: AgentAction, world: WorldRegistry, tick: int,
    ) -> BaseEvent | None:
        """Use-Action ausführen: Objekt muss im Raum sein, dann wirkt es auf die Stats."""
        room = world.get_room(self.profile.location_id)
        target = action.target or ""

        if target not in room.objects:
            self.profile.add_memory(
                f"Tick {tick}: Wollte {target or '?'} benutzen, aber das ist nicht hier"
            )
            logger.info("[%s] '%s' ist nicht im Raum", self.profile.name, target)
            return None

        self.profile.stats = apply_object_effect(self.profile.stats, target)
        self.profile.add_memory(f"Tick {tick}: Benutzte {target}")
        return BaseEvent(
            type=EventType.AGENT_USE,
            source=self.agent_id,
            location_id=self.profile.location_id,
            payload={"object": target},
        )

    def _execute_move(
        self, action: AgentAction, world: WorldRegistry, tick: int,
    ) -> BaseEvent | None:
        """Move-Action ausführen."""
        room = world.get_room(self.profile.location_id)
        target_dir = action.target or ""
        target_room_id = room.exits.get(target_dir)

        if target_room_id is None:
            self.profile.add_memory(
                f"Tick {tick}: Wollte nach {target_dir} gehen, aber kein Ausgang"
            )
            logger.info("[%s] Kein Ausgang Richtung '%s'", self.profile.name, target_dir)
            return None

        old_loc = self.profile.location_id
        world.move_agent(self.agent_id, old_loc, target_room_id)
        self.profile.location_id = target_room_id
        self.profile.add_memory(f"Tick {tick}: Ging von {old_loc} nach {target_room_id}")

        return BaseEvent(
            type=EventType.AGENT_MOVE,
            source=self.agent_id,
            location_id=old_loc,
            payload={"from": old_loc, "to": target_room_id},
        )
