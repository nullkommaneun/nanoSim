"""Konfiguration des Experiment-Labors: Experimente, Konstanten, Kommando-Bau.

Reine Daten/Funktionen — keine Seiteneffekte, voll unit-testbar.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo-Wurzel: …/src/nanosim/lab/config.py → 3 Ebenen hoch.
REPO = Path(__file__).resolve().parents[3]
# Ergebnis-Ordner (für Tests via Env überschreibbar).
RESULTS = Path(os.environ.get("LAB_RESULTS", str(REPO / "experiments" / "results")))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TICKS = 15
RUN_TIMEOUT = 1800  # harte Obergrenze pro Einzel-Lauf (s)
MAX_HOURS = 10  # Wanduhr-Sicherheitsstopp
TARGET_N = int(os.environ.get("LAB_TARGET_N", "60"))  # Ziel-Stichproben je Config
RANK_KEY = "reaction_chains"  # Headline-Kennzahl fürs Ranking

# Nachtsieger als feste Basis für die Stellschrauben-Experimente.
_WINNER = {"model": "llama3.2:3b", "dialogue": "llama3.1:8b"}

EXPERIMENTS: dict[str, list[dict]] = {
    "models": [
        {"name": "single-1b", "model": "gemma3:1b", "dialogue": None},
        {"name": "single-3b", "model": "llama3.2:3b", "dialogue": None},
        {"name": "single-8b", "model": "llama3.1:8b", "dialogue": None},
        {"name": "tier-1b+8b", "model": "gemma3:1b", "dialogue": "llama3.1:8b"},
        {"name": "tier-3b+8b", "model": "llama3.2:3b", "dialogue": "llama3.1:8b"},
    ],
    "prompt": [
        {"name": f"prompt-{v}", **_WINNER, "prompt_variant": v}
        for v in ("baseline", "soft", "hard", "naming")
    ],
    "world": [
        {"name": f"world-{w}", **_WINNER, "world": w}
        for w in ("default", "corridor6", "hub5", "twin")
    ],
    "memory": [
        {"name": f"mem-{m}", **_WINNER, "memory_window": m} for m in (1, 3, 6, 10)
    ],
}

METRIC_LABELS = {
    "reaction_chains": "Reaktionen",
    "togetherness": "Zusammen",
    "avg_max_room": "Gruppe",
    "unique_message_ratio": "Vielfalt",
    "moves": "Bewegung",
    "speaks": "Reden",
    "uses": "Objekte",
}


def active_configs(choice: str | None = None) -> list[dict]:
    """Welche Configs gesweept werden. 'all' = Prompt+Welt+Gedächtnis.

    Wirft hart bei unbekanntem Namen (kein stiller Fallback — sonst läuft
    nachts stundenlang das falsche Experiment).
    """
    choice = choice if choice is not None else os.environ.get("LAB_EXPERIMENT", "all")
    if choice == "all":
        return EXPERIMENTS["prompt"] + EXPERIMENTS["world"] + EXPERIMENTS["memory"]
    if choice not in EXPERIMENTS:
        allowed = ", ".join(["all", *EXPERIMENTS])
        raise ValueError(f"Unbekanntes LAB_EXPERIMENT '{choice}'. Erlaubt: {allowed}")
    return EXPERIMENTS[choice]


def build_cmd(
    cfg: dict,
    trace_path: Path | str,
    *,
    python: str | None = None,
    ticks: int = TICKS,
    url: str = OLLAMA_URL,
    seed: int | None = None,
) -> list[str]:
    """Den nanosim-Aufruf für eine Config zusammenbauen (rein, ohne Seiteneffekt)."""
    cmd = [
        python or sys.executable,
        "-m",
        "nanosim.main",
        "--model",
        cfg["model"],
        "--ticks",
        str(ticks),
        "--url",
        url,
        "--trace",
        str(trace_path),
    ]
    if cfg.get("dialogue"):
        cmd += ["--dialogue-model", cfg["dialogue"]]
    if cfg.get("world"):
        cmd += ["--world", cfg["world"]]
    if cfg.get("prompt_variant"):
        cmd += ["--prompt-variant", cfg["prompt_variant"]]
    if cfg.get("memory_window") is not None:
        cmd += ["--memory-window", str(cfg["memory_window"])]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd
