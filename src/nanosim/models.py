"""Zentrale Pydantic-Modelle für NanoSim-Pet."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Alle Event-Typen im System."""
    AGENT_SPEAK = "agent_speak"
    AGENT_MOVE = "agent_move"
    AGENT_USE = "agent_use"
    AGENT_REST = "agent_rest"
    AGENT_IDLE = "agent_idle"
    SYSTEM = "system"
    TICK = "tick"


class BaseEvent(BaseModel):
    """Ein Event das über den Bus geschickt wird."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: EventType
    source: str                          # Agent-ID oder "system"
    target: str | None = None            # Spezifischer Empfänger, None = alle
    location_id: str | None = None       # Room-Filter, None = Broadcast
    payload: dict[str, Any] = {}
    causality_depth: int = 0             # Verhindert Endlos-Event-Ketten


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgentStats(BaseModel):
    """Bedürfnisse eines Agenten. Alle Werte normalisiert auf [0.0, 1.0]."""
    stamina: float = Field(default=1.0, ge=0.0, le=1.0)
    mood: float = Field(default=1.0, ge=0.0, le=1.0)
    hunger: float = Field(default=0.0, ge=0.0, le=1.0)


class ActionType(str, Enum):
    """Mögliche Aktionen die ein Agent ausführen kann."""
    SPEAK = "speak"
    MOVE = "move"
    USE = "use"
    REST = "rest"
    IDLE = "idle"


class AgentAction(BaseModel):
    """Das Antwort-Schema das vom LLM zurückkommt."""
    action: ActionType
    target: str | None = None        # Room-Exit, Objekt-Name, oder None
    message: str | None = None       # Für speak-Actions


class AgentProfile(BaseModel):
    """Vollständiges Profil eines Agenten."""
    agent_id: str
    name: str
    persona: str                      # System-Prompt-Fragment
    location_id: str                  # Aktueller Room
    stats: AgentStats = Field(default_factory=AgentStats)
    inventory: list[str] = []
    memory: list[str] = []            # Rolling-List, max 10 Einträge

    def add_memory(self, entry: str, max_entries: int = 10) -> None:
        """Füge eine Erinnerung hinzu (FIFO wenn voll)."""
        self.memory.append(entry)
        if len(self.memory) > max_entries:
            self.memory = self.memory[-max_entries:]


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class Room(BaseModel):
    """Ein Raum im Terrarium."""
    room_id: str
    name: str
    description: str = ""
    occupants: set[str] = set()       # Agent-IDs
    objects: list[str] = []           # Interagierbare Objekte
    exits: dict[str, str] = {}        # {"north": "garden", "east": "bedroom"}
