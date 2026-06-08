"""Lebendigkeits-Kennzahlen: misst aus einem Trace, wie lebendig ein Lauf war.

Genutzt vom Experiment-Labor (experiments/lab.py), aber als eigenständige,
testbare Analyse-Funktion Teil des Pakets.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, stdev

from nanosim.models import ActionType, EventType
from nanosim.trace import Trace


def compute_metrics(trace: Trace) -> dict[str, float]:
    """Aus einem Trace die Kennzahlen eines Laufs berechnen.

    - reaction_chains: Reden, die echte Antworten auf jemanden sind (caused_by)
    - speaks / moves / uses: Aktions-Zähler
    - togetherness: Anteil der Ticks mit >=2 Tieren im selben Raum
    - avg_max_room: durchschnittliche Größe der vollsten Raum-Gruppe
    - unique_message_ratio: Anteil eindeutiger Reden (1.0 = nie Wiederholung)
    """
    ticks = trace.ticks
    n_ticks = len(ticks) or 1

    reaction_chains = moves = uses = 0
    messages: list[str] = []
    together_ticks = 0
    max_occ_sum = 0

    for tick in ticks:
        for d in tick.decisions:
            if d.action == ActionType.MOVE:
                moves += 1
            elif d.action == ActionType.USE:
                uses += 1
            elif d.action == ActionType.SPEAK and d.message:
                messages.append(d.message.strip())

        for e in tick.events:
            if e.type == EventType.AGENT_SPEAK and e.caused_by:
                reaction_chains += 1

        rooms = Counter(s.location_id for s in tick.agents)
        max_occ = max(rooms.values()) if rooms else 0
        max_occ_sum += max_occ
        if max_occ >= 2:
            together_ticks += 1

    n_speaks = len(messages)
    unique_ratio = len(set(messages)) / n_speaks if n_speaks else 0.0

    return {
        "ticks": len(ticks),
        "reaction_chains": reaction_chains,
        "speaks": n_speaks,
        "moves": moves,
        "uses": uses,
        "togetherness": together_ticks / n_ticks,
        "avg_max_room": max_occ_sum / n_ticks,
        "unique_message_ratio": unique_ratio,
    }


def aggregate(metrics_list: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mehrere Mess-Ergebnisse zu Mittelwert + Streuung je Kennzahl verdichten."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        out[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return out
