"""Report-Rendering — ehrlich: Erfolgsquote, Konfidenzintervalle, Top-Gruppe.

Reine Funktion: Records (+ Metadaten) rein, Markdown raus.
"""

from __future__ import annotations

from nanosim.lab.config import METRIC_LABELS, RANK_KEY, TICKS
from nanosim.lab.record import RunRecord
from nanosim.lab.stats import RankedEntry, rank_configs

_EXTRA_COLS = ["togetherness", "avg_max_room", "unique_message_ratio", "moves"]


def _row(rank: int, e: RankedEntry) -> str:
    headline = f"{e.mean:.2f} ± {e.std:.2f} [{e.ci_lo:.1f}–{e.ci_hi:.1f}]"
    extras = " | ".join(f"{e.extra.get(k, 0.0):.2f}" for k in _EXTRA_COLS)
    tie = " ＝" if e.tied_with_top and rank > 1 else ""
    return (
        f"| {rank}{tie} | `{e.name}` | {e.n_ok}/{e.n_total} | {headline} | {extras} |"
    )


def render_report(records: list[RunRecord], *, meta: dict | None = None) -> str:
    meta = meta or {}
    ranked, degenerate = rank_configs(records, rank_key=RANK_KEY)
    n_total = len(records)
    n_ok = sum(1 for r in records if r.ok)
    n_fail = n_total - n_ok

    health = meta.get("health", "—")
    lines = [
        "# nanoSim Experiment-Labor — Report",
        "",
        f"Stand: {meta.get('timestamp', '—')} · Experiment: **{meta.get('experiment', '—')}** · "
        f"Status: **{health}** · Commit: `{meta.get('commit', '—')}`",
        f"Läufe: {n_ok} ok / {n_fail} fehlgeschlagen ({n_total} gesamt) · "
        f"{TICKS} Ticks/Lauf · Rest-Budget: {meta.get('budget_left', '—')}",
        "",
        "| Rang | Konfiguration | n (ok/ges.) | "
        + METRIC_LABELS[RANK_KEY]
        + " [95%-KI] | "
        + " | ".join(METRIC_LABELS[k] for k in _EXTRA_COLS)
        + " |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rank, e in enumerate(ranked, 1):
        lines.append(_row(rank, e))

    # Top-Gruppe statt Einzel-Sieger, wenn die KIs überlappen.
    if ranked:
        top_group = [e.name for e in ranked if e.tied_with_top]
        if len(top_group) > 1:
            lines += [
                "",
                "🏁 **Top-Gruppe (statistisch gleichauf):** "
                + ", ".join(f"`{n}`" for n in top_group),
            ]
        else:
            lines += ["", f"🏆 **Bester Aufbau:** `{ranked[0].name}`"]

    if degenerate:
        lines += [
            "",
            "⚠️ **Aus dem Ranking genommen (degeneriert, "
            f"{METRIC_LABELS[RANK_KEY]}=0 — Pipeline-Defekt, keine Interaktion):** "
            + ", ".join(f"`{e.name}` ({e.n_ok}/{e.n_total} ok)" for e in degenerate),
        ]

    lines += [
        "",
        "**Kennzahlen:** Reaktionen = echte Antworten auf jemanden (mit 95%-Bootstrap-KI) · "
        "Zusammen = Anteil Ticks mit ≥2 Tieren im selben Raum · Gruppe = ø größte Raum-Gruppe · "
        "Vielfalt = Anteil eindeutiger Reden · Bewegung = Raumwechsel. "
        "`＝` = Konfidenzintervall überlappt den Spitzenreiter (nicht unterscheidbar).",
    ]
    return "\n".join(lines) + "\n"
