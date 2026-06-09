"""Datensatz eines Einzel-Laufs + fehlertolerantes Einlesen.

Eine einzige kaputte Zeile (z.B. nach Stromausfall) darf NIE den Report oder
den Sweep blockieren — sie wird übersprungen und gezählt.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RunRecord(BaseModel):
    """Ein protokollierter Lauf (Erfolg ODER Fehlschlag)."""

    round: int
    config: str
    ts: float
    status: str = "ok"  # ok | timeout | oom | error | skip
    metrics: dict = Field(default_factory=dict)
    duration_s: float | None = None
    seed: int | None = None
    # Config-Felder (für Nachvollziehbarkeit mitgeschrieben)
    model: str | None = None
    dialogue: str | None = None
    world: str | None = None
    prompt_variant: str | None = None
    memory_window: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def load_records(path: Path | str) -> tuple[list[RunRecord], int]:
    """Alle Records laden; (records, anzahl_übersprungener_kaputter_zeilen)."""
    p = Path(path)
    records: list[RunRecord] = []
    skipped = 0
    if not p.exists():
        return records, 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(RunRecord.model_validate_json(line))
        except Exception:  # noqa: BLE001 — kaputte Zeile überspringen, nie crashen
            skipped += 1
    return records, skipped


def append_record(path: Path | str, record: RunRecord) -> None:
    """Einen Record atomar-ish als JSON-Zeile anhängen (sofort geflusht)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
        f.flush()
