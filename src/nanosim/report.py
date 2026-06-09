"""HTML-Report: aus einem Trace eine eigenständige, offline lauffähige Datei.

Erzeugt EINE HTML-Datei mit eingebetteter Plotly-Bibliothek (kein Server,
kein CDN, per Doppelklick im Browser). Vier Abschnitte:

- Lebenskurven   : Energie/Stimmung/Hunger jedes Tiers über die Ticks
- Reaktionsketten: der Causality-Baum der Gespräche (nutzt caused_by)
- Raumwechsel    : wer war wann in welchem Raum
- Tick-Protokoll : die lesbare Geschichte des Laufs

Plotly ist eine optionale Abhängigkeit: pip install -e .[report]
"""

from __future__ import annotations

import html
from pathlib import Path

from nanosim.models import EventType
from nanosim.replay import render_trace_lines
from nanosim.trace import Trace, load_trace

# Farbpalette pro Agent (gut unterscheidbar)
_PALETTE = ["#e6550d", "#3182bd", "#31a354", "#756bb1", "#636363", "#d6616b"]
# Linienstil pro Stat
_STAT_DASH = {"Energie": "solid", "Stimmung": "dot", "Hunger": "dash"}


def _require_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - nur ohne Plotly
        raise RuntimeError(
            "Der HTML-Report braucht Plotly. Installiere es mit:\n"
            "  pip install -e .[report]"
        ) from exc
    return go, pio


def _agent_series(trace: Trace) -> dict[str, dict]:
    """Pro Agent die Zeitreihen (Ticks, Stats, Raum) einsammeln."""
    series: dict[str, dict] = {}
    for tick in trace.ticks:
        for snap in tick.agents:
            s = series.setdefault(
                snap.agent_id,
                {
                    "name": snap.name,
                    "ticks": [],
                    "Energie": [],
                    "Stimmung": [],
                    "Hunger": [],
                    "rooms": [],
                },
            )
            s["ticks"].append(tick.tick)
            s["Energie"].append(snap.stats.stamina)
            s["Stimmung"].append(snap.stats.mood)
            s["Hunger"].append(snap.stats.hunger)
            s["rooms"].append(snap.location_id)
    return series


def _stat_figure(trace: Trace):
    go, _ = _require_plotly()
    fig = go.Figure()
    series = _agent_series(trace)
    for i, (_aid, s) in enumerate(series.items()):
        color = _PALETTE[i % len(_PALETTE)]
        for stat, dash in _STAT_DASH.items():
            fig.add_trace(
                go.Scatter(
                    x=s["ticks"],
                    y=s[stat],
                    name=f"{s['name']} · {stat}",
                    mode="lines+markers",
                    line={"color": color, "dash": dash},
                )
            )
    fig.update_layout(
        title="Lebenskurven",
        xaxis_title="Tick",
        yaxis_title="Wert (0–1)",
        yaxis={"range": [0, 1]},
        template="plotly_white",
        height=420,
    )
    return fig


def _causality_figure(trace: Trace):
    go, _ = _require_plotly()
    ids, parents, labels = [], [], []
    for tick in trace.ticks:
        for ev in tick.events:
            if ev.type != EventType.AGENT_SPEAK:
                continue
            msg = ev.payload.get("message", "...")
            ids.append(ev.id)
            parents.append(ev.caused_by or "")
            labels.append(f"{ev.source}: {msg}")

    if not ids:
        fig = go.Figure()
        fig.add_annotation(
            text="Keine Gespräche in diesem Lauf", showarrow=False, font={"size": 16}
        )
        fig.update_layout(title="Reaktionsketten", height=300, template="plotly_white")
        return fig

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            branchvalues="remainder",
            root_color="lightgrey",
        )
    )
    fig.update_layout(
        title="Reaktionsketten (wer reagiert auf wen)",
        height=420,
        template="plotly_white",
    )
    return fig


def _movement_figure(trace: Trace):
    go, _ = _require_plotly()
    fig = go.Figure()
    series = _agent_series(trace)
    for i, (_aid, s) in enumerate(series.items()):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=s["ticks"],
                y=s["rooms"],
                name=s["name"],
                mode="lines+markers",
                line={"color": color, "shape": "hv"},
            )
        )
    fig.update_layout(
        title="Raumwechsel",
        xaxis_title="Tick",
        yaxis_title="Raum",
        template="plotly_white",
        height=360,
    )
    return fig


_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>nanoSim — Lauf-Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 1.5rem auto;
          padding: 0 1rem; color: #222; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  h2 {{ border-bottom: 2px solid #eee; padding-bottom: 0.3rem; margin-top: 2rem; }}
  pre {{ background: #f6f8fa; padding: 1rem; border-radius: 8px; overflow-x: auto;
         line-height: 1.4; }}
</style>
</head>
<body>
<h1>🐾 nanoSim — Lauf-Report</h1>
<p class="meta">Modell: {model} · {n} Ticks</p>

<h2>Lebenskurven</h2>
{fig_stats}

<h2>Reaktionsketten</h2>
{fig_causality}

<h2>Raumwechsel</h2>
{fig_movement}

<h2>Tick-Protokoll</h2>
<pre>{protocol}</pre>
</body>
</html>
"""


def build_report_html(trace: Trace) -> str:
    """Den kompletten, eigenständigen HTML-Report als String erzeugen."""
    _, pio = _require_plotly()

    # Erste Figur bettet die Plotly-Bibliothek ein, die anderen nutzen sie mit.
    fig_stats = pio.to_html(_stat_figure(trace), include_plotlyjs=True, full_html=False)
    fig_causality = pio.to_html(
        _causality_figure(trace), include_plotlyjs=False, full_html=False
    )
    fig_movement = pio.to_html(
        _movement_figure(trace), include_plotlyjs=False, full_html=False
    )

    protocol = html.escape("\n".join(render_trace_lines(trace)))

    return _PAGE.format(
        model=html.escape(trace.meta.model),
        n=len(trace.ticks),
        fig_stats=fig_stats,
        fig_causality=fig_causality,
        fig_movement=fig_movement,
        protocol=protocol,
    )


def write_report(trace_path: str | Path, out_path: str | Path) -> None:
    """Einen Trace laden und als HTML-Report auf die Platte schreiben."""
    trace = load_trace(trace_path)
    Path(out_path).write_text(build_report_html(trace), encoding="utf-8")
