"""Replay/Playback: einen aufgezeichneten Lauf wieder abspielen — ohne LLM.

Das ist die ehrliche Antwort auf "Reproduzierbarkeit": Statt zu versuchen,
das LLM deterministisch zu zwingen (auf Ollama praktisch unmöglich), spielen
wir die *aufgezeichnete* Spur exakt so wieder ab, wie sie passiert ist —
offline, sofort, ohne Ollama. Faithful by construction.
"""

from __future__ import annotations

from pathlib import Path

from nanosim.models import BaseEvent, EventType
from nanosim.trace import TickRecord, Trace, load_trace


def format_event(event: BaseEvent) -> str:
    """Ein Event in eine lesbare Zeile übersetzen."""
    src = event.source
    p = event.payload
    if event.type == EventType.AGENT_SPEAK:
        return f'  💬 {src} sagt: "{p.get("message", "...")}"'
    if event.type == EventType.AGENT_MOVE:
        return f"  🚶 {src} geht von {p.get('from', '?')} nach {p.get('to', '?')}"
    if event.type == EventType.AGENT_USE:
        return f"  🐾 {src} benutzt {p.get('object', '?')}"
    if event.type == EventType.AGENT_REST:
        return f"  😴 {src} ruht sich aus"
    return f"  · {src}: {event.type.value}"


def summarize_tick(tick: TickRecord) -> list[str]:
    """Einen Tick in lesbare Zeilen übersetzen (Header + Ereignisse + Stats)."""
    lines = [f"── TICK {tick.tick} " + "─" * 30]

    if tick.events:
        lines.extend(format_event(ev) for ev in tick.events)
    else:
        lines.append("  (still — niemand tut etwas Sichtbares)")

    for snap in tick.agents:
        s = snap.stats
        lines.append(
            f"     {snap.name} @ {snap.location_id} | "
            f"Energie={s.stamina:.2f} Stimmung={s.mood:.2f} Hunger={s.hunger:.2f}"
        )
    return lines


def render_trace_lines(trace: Trace) -> list[str]:
    """Den gesamten Trace in lesbare Zeilen übersetzen."""
    lines = [f"Replay: Modell={trace.meta.model} | {len(trace.ticks)} Ticks", ""]
    for tick in trace.ticks:
        lines.extend(summarize_tick(tick))
        lines.append("")
    return lines


def play_trace(path: str | Path) -> None:
    """Eine Trace-Datei laden und Tick für Tick auf der Konsole abspielen."""
    from rich.console import Console

    console = Console()
    trace = load_trace(path)
    for line in render_trace_lines(trace):
        console.print(line)
