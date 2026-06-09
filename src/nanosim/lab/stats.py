"""Ehrliche Auswertung: Bootstrap-Konfidenzintervalle statt blanker Mittelwerte,
Erkennung degenerierter Configs, Erfolgsquoten.

Reine Funktionen — deterministisch (fester RNG-Seed), voll testbar.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean, stdev

from nanosim.lab.record import RunRecord


@dataclass
class RankedEntry:
    name: str
    n_ok: int
    n_total: int
    mean: float
    std: float
    ci_lo: float
    ci_hi: float
    degenerate: bool = False
    tied_with_top: bool = False
    extra: dict = field(default_factory=dict)  # ø weiterer Kennzahlen

    @property
    def success_rate(self) -> float:
        return self.n_ok / self.n_total if self.n_total else 0.0


def group_by_config(records: list[RunRecord]) -> dict[str, list[RunRecord]]:
    out: dict[str, list[RunRecord]] = {}
    for r in records:
        out.setdefault(r.config, []).append(r)
    return out


def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Perzentil-Bootstrap-Konfidenzintervall des Mittelwerts (deterministisch)."""
    if not values:
        return (0.0, 0.0)
    if len(values) < 2:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        mean(rng.choice(values) for _ in range(n)) for _ in range(n_boot)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(alpha * n_boot)]
    hi = means[min(n_boot - 1, int((1.0 - alpha) * n_boot))]
    return (lo, hi)


def _ci_overlap(a: RankedEntry, b: RankedEntry) -> bool:
    return not (a.ci_hi < b.ci_lo or b.ci_hi < a.ci_lo)


def rank_configs(
    records: list[RunRecord],
    *,
    rank_key: str = "reaction_chains",
    extra_keys: tuple[str, ...] = (
        "togetherness",
        "avg_max_room",
        "unique_message_ratio",
        "moves",
    ),
) -> tuple[list[RankedEntry], list[RankedEntry]]:
    """Configs ranken. Gibt (ranking, degenerierte) zurück.

    - Degeneriert = alle erfolgreichen Läufe haben rank_key==0 (z.B. ein
      Pipeline-Defekt wie 1b, das nie Interaktion auslöst) → NICHT ranken.
    - Ranking nach Mittelwert; Configs, deren CI das des Spitzenreiters
      überlappt, werden als "gleichauf" markiert (kein falscher Einzel-Sieger).
    """
    ranked: list[RankedEntry] = []
    degenerate: list[RankedEntry] = []

    for name, runs in group_by_config(records).items():
        ok = [r for r in runs if r.ok and rank_key in r.metrics]
        values = [float(r.metrics[rank_key]) for r in ok]
        entry = RankedEntry(
            name=name,
            n_ok=len(ok),
            n_total=len(runs),
            mean=mean(values) if values else 0.0,
            std=stdev(values) if len(values) > 1 else 0.0,
            ci_lo=0.0,
            ci_hi=0.0,
        )
        if values:
            entry.ci_lo, entry.ci_hi = bootstrap_ci(values)
            entry.extra = {
                k: mean([float(r.metrics[k]) for r in ok if k in r.metrics] or [0.0])
                for k in extra_keys
            }
        if values and all(v == 0 for v in values):
            entry.degenerate = True
            degenerate.append(entry)
        else:
            ranked.append(entry)

    ranked.sort(key=lambda e: e.mean, reverse=True)
    if ranked:
        top = ranked[0]
        for e in ranked:
            e.tied_with_top = _ci_overlap(e, top)
    return ranked, degenerate
