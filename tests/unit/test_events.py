"""Tests für den EventBus."""

import asyncio

import pytest

from nanosim.core.events import EventBus
from nanosim.models import BaseEvent, EventType


@pytest.fixture
def bus():
    return EventBus()


def _make_event(source: str, location_id: str | None = None, target: str | None = None):
    return BaseEvent(
        type=EventType.AGENT_SPEAK,
        source=source,
        location_id=location_id,
        target=target,
        payload={"message": f"Nachricht von {source}"},
    )


class TestEventBus:
    @pytest.mark.asyncio
    async def test_broadcast_event(self, bus: EventBus):
        """Events ohne location_id gehen an alle (außer Source)."""
        received_a: list[BaseEvent] = []
        received_b: list[BaseEvent] = []

        bus.subscribe("agent_a", lambda e: _append(received_a, e), lambda: "kitchen")
        bus.subscribe("agent_b", lambda e: _append(received_b, e), lambda: "garden")

        await bus.publish(_make_event("system"))
        await bus.drain()

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_location_filter(self, bus: EventBus):
        """Events mit location_id nur an Agenten im selben Room."""
        received_a: list[BaseEvent] = []
        received_b: list[BaseEvent] = []

        bus.subscribe("agent_a", lambda e: _append(received_a, e), lambda: "kitchen")
        bus.subscribe("agent_b", lambda e: _append(received_b, e), lambda: "garden")

        await bus.publish(_make_event("system", location_id="kitchen"))
        await bus.drain()

        assert len(received_a) == 1
        assert len(received_b) == 0

    @pytest.mark.asyncio
    async def test_source_excluded(self, bus: EventBus):
        """Source-Agent bekommt sein eigenes Event nicht."""
        received: list[BaseEvent] = []

        bus.subscribe("agent_a", lambda e: _append(received, e), lambda: "kitchen")
        bus.subscribe("agent_b", lambda e: _append(received, e), lambda: "kitchen")

        await bus.publish(_make_event("agent_a", location_id="kitchen"))
        await bus.drain()

        assert len(received) == 1
        assert received[0].payload["message"] == "Nachricht von agent_a"

    @pytest.mark.asyncio
    async def test_target_filter(self, bus: EventBus):
        """Events mit target nur an den spezifischen Empfänger."""
        received_a: list[BaseEvent] = []
        received_b: list[BaseEvent] = []

        bus.subscribe("agent_a", lambda e: _append(received_a, e), lambda: "kitchen")
        bus.subscribe("agent_b", lambda e: _append(received_b, e), lambda: "kitchen")

        await bus.publish(_make_event("system", location_id="kitchen", target="agent_a"))
        await bus.drain()

        assert len(received_a) == 1
        assert len(received_b) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus: EventBus):
        received: list[BaseEvent] = []
        bus.subscribe("agent_a", lambda e: _append(received, e), lambda: "kitchen")
        bus.unsubscribe("agent_a")

        await bus.publish(_make_event("system"))
        await bus.drain()

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_events_ordering(self, bus: EventBus):
        """Events werden in der Reihenfolge zugestellt wie sie published wurden."""
        received: list[BaseEvent] = []
        bus.subscribe("agent_a", lambda e: _append(received, e), lambda: "kitchen")

        for i in range(5):
            await bus.publish(BaseEvent(
                type=EventType.TICK,
                source="system",
                payload={"tick": i},
            ))
        await bus.drain()

        assert [e.payload["tick"] for e in received] == [0, 1, 2, 3, 4]


async def _append(lst: list, event: BaseEvent) -> None:
    lst.append(event)
