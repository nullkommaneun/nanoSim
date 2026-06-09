#!/usr/bin/env python3
"""nanoSim Experiment-Labor — autonomer Nacht-Sweep.

Probiert Runde um Runde verschiedene Konfigurationen durch, misst pro Lauf die
Lebendigkeits-Kennzahlen (nanosim.metrics) und schreibt nach jeder Runde einen
aktualisierten Report. Läuft, bis die Zeit-Obergrenze erreicht ist oder das
Skript beendet wird — die Ergebnisse sammeln sich inkrementell an.

Robust für unbeaufsichtigten Betrieb:
- Ollama wird bei Bedarf automatisch (neu)gestartet.
- Jeder Lauf läuft in einem eigenen Prozess mit Timeout; ein Fehler killt nicht
  den Sweep, sondern wird protokolliert.
- Jedes Ergebnis wird sofort als JSONL angehängt (überlebt einen Abbruch).

Start (aus dem Repo, mit aktivem venv):
    nohup python experiments/lab.py > experiments/results/nohup.out 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from nanosim.metrics import aggregate, compute_metrics
from nanosim.trace import load_trace

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
RUNS_FILE = RESULTS / "runs.jsonl"
REPORT_FILE = RESULTS / "report.md"
LOG_FILE = RESULTS / "lab.log"

OLLAMA_URL = "http://localhost:11434"
TICKS = 15
RUN_TIMEOUT = 1800  # max 30 min pro Einzel-Lauf
MAX_HOURS = 10  # Sicherheits-Stopp
RANK_KEY = "reaction_chains"  # Headline-Kennzahl fürs Ranking

# Konfigurationen, die verglichen werden (Name, Entscheidungs-Modell, Dialog-Modell)
CONFIGS = [
    {"name": "single-1b", "model": "gemma3:1b", "dialogue": None},
    {"name": "single-3b", "model": "llama3.2:3b", "dialogue": None},
    {"name": "single-8b", "model": "llama3.1:8b", "dialogue": None},
    {"name": "tier-1b+8b", "model": "gemma3:1b", "dialogue": "llama3.1:8b"},
    {"name": "tier-3b+8b", "model": "llama3.2:3b", "dialogue": "llama3.1:8b"},
]

METRIC_LABELS = {
    "reaction_chains": "Reaktionen",
    "togetherness": "Zusammen",
    "avg_max_room": "Gruppe",
    "unique_message_ratio": "Vielfalt",
    "moves": "Bewegung",
    "speaks": "Reden",
    "uses": "Objekte",
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def ensure_ollama() -> bool:
    """Stelle sicher, dass der Ollama-Server läuft (sonst starten + warten)."""
    if ollama_up():
        return True
    log("Ollama nicht erreichbar — starte 'ollama serve' ...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        log("FEHLER: 'ollama' nicht gefunden.")
        return False
    for _ in range(30):
        time.sleep(1)
        if ollama_up():
            log("Ollama ist wieder da.")
            return True
    log("FEHLER: Ollama kam nicht hoch.")
    return False


def run_one(cfg: dict, idx: int) -> dict | None:
    """Eine Simulation als eigener Prozess, dann Kennzahlen aus dem Trace."""
    trace_path = RESULTS / f"_tmp_{cfg['name']}_{idx}.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "nanosim.main",
        "--model",
        cfg["model"],
        "--ticks",
        str(TICKS),
        "--trace",
        str(trace_path),
    ]
    if cfg["dialogue"]:
        cmd += ["--dialogue-model", cfg["dialogue"]]
    try:
        subprocess.run(
            cmd,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=RUN_TIMEOUT,
            check=True,
        )
        metrics = compute_metrics(load_trace(trace_path))
        return metrics
    except Exception as e:  # noqa: BLE001 — ein Lauf darf scheitern, ohne alles zu stoppen
        log(f"  Lauf fehlgeschlagen ({cfg['name']}): {type(e).__name__}: {e}")
        return None
    finally:
        trace_path.unlink(missing_ok=True)


def write_report() -> None:
    """Alle bisherigen Läufe einlesen, je Konfiguration aggregieren, Report schreiben."""
    if not RUNS_FILE.exists():
        return
    by_config: dict[str, list[dict]] = {}
    for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_config.setdefault(rec["config"], []).append(rec["metrics"])

    ranked = sorted(
        by_config.items(),
        key=lambda kv: aggregate(kv[1]).get(RANK_KEY, {}).get("mean", 0),
        reverse=True,
    )

    total = sum(len(v) for v in by_config.values())
    lines = [
        "# nanoSim Experiment-Labor — Report",
        "",
        f"Stand: {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"{total} Läufe · {TICKS} Ticks/Lauf · Ranking nach **{METRIC_LABELS[RANK_KEY]}**",
        "",
        "| Rang | Konfiguration | n | "
        + " | ".join(
            METRIC_LABELS[k]
            for k in [
                "reaction_chains",
                "togetherness",
                "avg_max_room",
                "unique_message_ratio",
                "moves",
            ]
        )
        + " |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rank, (name, runs) in enumerate(ranked, 1):
        agg = aggregate(runs)

        def cell(key, agg=agg):
            a = agg[key]
            return f"{a['mean']:.2f} ± {a['std']:.2f}"

        lines.append(
            f"| {rank} | `{name}` | {len(runs)} | "
            + " | ".join(
                cell(k)
                for k in [
                    "reaction_chains",
                    "togetherness",
                    "avg_max_room",
                    "unique_message_ratio",
                    "moves",
                ]
            )
            + " |"
        )

    lines += [
        "",
        "**Kennzahlen:** Reaktionen = echte Antworten auf jemanden · "
        "Zusammen = Anteil Ticks mit ≥2 Tieren im selben Raum · "
        "Gruppe = ø größte Raum-Gruppe · Vielfalt = Anteil eindeutiger Reden "
        "(1.0 = nie Wiederholung) · Bewegung = Anzahl Raumwechsel.",
    ]
    if ranked:
        best = ranked[0][0]
        lines += ["", f"🏆 **Bestes Setup bisher:** `{best}`"]

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log(
        f"=== Experiment-Labor gestartet (max {MAX_HOURS}h, {len(CONFIGS)} Konfigs) ==="
    )
    start = time.time()
    round_i = 0
    while time.time() - start < MAX_HOURS * 3600:
        round_i += 1
        log(f"--- Runde {round_i} ---")
        for cfg in CONFIGS:
            if not ensure_ollama():
                time.sleep(30)
                continue
            metrics = run_one(cfg, round_i)
            if metrics is None:
                continue
            rec = {
                "round": round_i,
                "config": cfg["name"],
                "model": cfg["model"],
                "dialogue": cfg["dialogue"],
                "ts": time.time(),
                "metrics": metrics,
            }
            with RUNS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log(
                f"  {cfg['name']}: {RANK_KEY}={metrics[RANK_KEY]} "
                f"together={metrics['togetherness']:.2f} "
                f"vielfalt={metrics['unique_message_ratio']:.2f}"
            )
        write_report()
        log(f"Runde {round_i} fertig, Report aktualisiert.")
    log("=== Zeit-Obergrenze erreicht, Labor beendet. ===")


if __name__ == "__main__":
    main()
