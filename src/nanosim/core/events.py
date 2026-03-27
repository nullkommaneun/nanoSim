"""Event-Bus mit Location-basierter Filterung."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from nanosim.models import BaseEvent


# Typ für async Event-Handler
EventHandler = Callable[[BaseEvent], Coroutine[Any, Any, None]]


class EventBus:
    """Einfacher asyncio-basierter Pub/Sub mit Room-Filterung.

    Subscriber registrieren sich mit einer agent_id und einer location-Funktion.
    Events mit location_id werden nur an Subscriber im selben Room zugestellt.
    Events ohne location_id sind Broadcasts (System-Events).
    """

    def __init__(self) -> None:
        # {agent_id: (handler, location_fn)}
        self._subscribers: dict[str, tuple[EventHandler, Callable[[], str | None]]] = {}
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue()

    def subscribe(
        self,
        agent_id: str,
        handler: EventHandler,
        location_fn: Callable[[], str | None],
    ) -> None:
        """Registriere einen Subscriber mit seiner Location-Funktion."""
        self._subscribers[agent_id] = (handler, location_fn)

    def unsubscribe(self, agent_id: str) -> None:
        self._subscribers.pop(agent_id, None)

    async def publish(self, event: BaseEvent) -> None:
        """Event in die Queue legen."""
        await self._queue.put(event)

    async def drain(self) -> None:
        """Alle Events in der Queue verarbeiten und zustellen."""
        while not self._queue.empty():
            event = self._queue.get_nowait()
            await self._deliver(event)

    async def _deliver(self, event: BaseEvent) -> None:
        """Ein einzelnes Event an die passenden Subscriber zustellen."""
        for agent_id, (handler, location_fn) in self._subscribers.items():
            # Source-Agent bekommt eigene Events nicht nochmal
            if agent_id == event.source:
                continue

            # Location-Filter: Event mit location_id nur an Agenten im selben Room
            if event.location_id is not None:
                agent_location = location_fn()
                if agent_location != event.location_id:
                    continue

            # Target-Filter: wenn ein spezifischer Empfänger gesetzt ist
            if event.target is not None and event.target != agent_id:
                continue

            await handler(event)
