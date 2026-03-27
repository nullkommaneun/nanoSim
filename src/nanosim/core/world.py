"""World-Registry: Verwaltet Rooms und Agent-Positionen."""

from __future__ import annotations

from nanosim.models import Room


class WorldRegistry:
    """Zentrale Registry für alle Rooms im Terrarium."""

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}

    def add_room(self, room: Room) -> None:
        self._rooms[room.room_id] = room

    def get_room(self, room_id: str) -> Room:
        return self._rooms[room_id]

    def all_rooms(self) -> list[Room]:
        return list(self._rooms.values())

    def agents_in_room(self, room_id: str) -> set[str]:
        return self._rooms[room_id].occupants

    def move_agent(self, agent_id: str, from_room_id: str, to_room_id: str) -> None:
        """Bewege einen Agenten von einem Room in einen anderen."""
        from_room = self._rooms[from_room_id]
        to_room = self._rooms[to_room_id]
        from_room.occupants.discard(agent_id)
        to_room.occupants.add(agent_id)
