"""Trace-Aufzeichnung: Jeder Lauf wird als JSONL mitgeschrieben.

Format (eine JSON-Zeile pro Record):
  Zeile 1   : TraceMeta  (Lauf-Metadaten)
  Zeile 2..n: TickRecord (ein Record pro Tick)

Die Felder lehnen sich an die OpenTelemetry-GenAI-Namenskonventionen an
(z. B. `model` ~ `gen_ai.request.model`), damit der Trace sich später ohne
Neubau in Standard-Tools (Langfuse, Arize Phoenix, ...) überführen lässt.

Der Trace zeichnet auch die getroffene Entscheidung (`Decision`) pro Agent
und Tick auf — das ist die Grundlage für den deterministischen Replay-Modus
(Nachspielen ohne LLM).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from nanosim.models import ActionType, AgentStats, BaseEvent

TRACE_SCHEMA_VERSION = "nanosim.trace.v1"


class TraceMeta(BaseModel):
    """Metadaten eines Laufs (erste Zeile der Trace-Datei)."""

    schema_version: str = TRACE_SCHEMA_VERSION
    model: str
    agent_ids: list[str] = Field(default_factory=list)


class AgentSnapshot(BaseModel):
    """Momentaufnahme eines Agenten zu einem Tick."""

    agent_id: str
    name: str
    location_id: str
    stats: AgentStats


class Decision(BaseModel):
    """Die vom Agenten (via LLM) getroffene Entscheidung — Basis für Replay."""

    agent_id: str
    action: ActionType
    target: str | None = None
    message: str | None = None


class TickRecord(BaseModel):
    """Vollständiger Mitschnitt eines einzelnen Ticks."""

    tick: int
    agents: list[AgentSnapshot] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    events: list[BaseEvent] = Field(default_factory=list)


class Trace(BaseModel):
    """Ein vollständig geladener Lauf: Metadaten + alle Ticks."""

    meta: TraceMeta
    ticks: list[TickRecord] = Field(default_factory=list)


class TraceWriter:
    """Schreibt einen Lauf inkrementell als JSONL auf die Platte.

    Die Meta-Zeile wird sofort beim Öffnen geschrieben, jeder Tick beim
    Aufruf von `record_tick`. So überlebt der Trace auch einen Abbruch.
    """

    def __init__(
        self, path: str | Path, *, model: str, agent_ids: list[str],
    ) -> None:
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")
        meta = TraceMeta(model=model, agent_ids=list(agent_ids))
        self._write_line(meta.model_dump_json())

    def record_tick(self, record: TickRecord) -> None:
        """Einen Tick-Record als JSON-Zeile anhängen."""
        self._write_line(record.model_dump_json())

    def close(self) -> None:
        """Datei schließen."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write_line(self, line: str) -> None:
        self._fh.write(line + "\n")
        self._fh.flush()


def load_trace(path: str | Path) -> Trace:
    """Eine JSONL-Trace-Datei laden: erste Zeile = Meta, Rest = Ticks."""
    lines = [
        ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    if not lines:
        raise ValueError(f"Leere Trace-Datei: {path}")

    meta = TraceMeta.model_validate_json(lines[0])
    ticks = [TickRecord.model_validate_json(ln) for ln in lines[1:]]
    return Trace(meta=meta, ticks=ticks)
