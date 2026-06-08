"""NanoSim-Pet Einstiegspunkt — Startet das Terrarium."""

from __future__ import annotations

import argparse
import asyncio
import logging

from rich.console import Console
from rich.logging import RichHandler

from nanosim.agents.base import BaseAgent
from nanosim.core.events import EventBus
from nanosim.core.tick import TickEngine
from nanosim.core.world import WorldRegistry
from nanosim.llm.router import DEFAULT_MODEL, LlamaRouter
from nanosim.models import AgentProfile
from nanosim.persistence import (
    load_snapshot,
    save_snapshot,
    world_from_snapshot,
)
from nanosim.replay import play_trace
from nanosim.report import write_report
from nanosim.trace import TraceWriter
from nanosim.world.personas import create_default_agents
from nanosim.world.rooms import create_default_world

console = Console()


def _build_world_and_profiles(
    load_path: str | None,
) -> tuple[WorldRegistry, list[AgentProfile], int]:
    """Welt + Agent-Profile bereitstellen.

    Mit `load_path` wird aus einem Snapshot fortgesetzt (inkl. Tick-Zähler),
    sonst eine frische Default-Welt gebaut und die Agenten platziert.
    """
    if load_path:
        snapshot = load_snapshot(load_path)
        world = world_from_snapshot(snapshot)
        return world, snapshot.profiles, snapshot.tick_count

    world = create_default_world()
    profiles = create_default_agents()
    for profile in profiles:
        world.get_room(profile.location_id).occupants.add(profile.agent_id)
    return world, profiles, 0


def setup_logging(level: int = logging.INFO) -> None:
    """Konfiguriere Rich-Logging für hübsche Konsolenausgabe."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


async def run_terrarium(
    model: str = DEFAULT_MODEL,
    num_ticks: int = 5,
    base_url: str = "http://localhost:11434",
    load_path: str | None = None,
    save_path: str | None = None,
    trace_path: str | None = None,
) -> None:
    """Starte eine NanoSim-Pet Simulation.

    Args:
        model: Ollama-Modellname.
        num_ticks: Anzahl Ticks die simuliert werden.
        base_url: Ollama-Server URL.
        load_path: Optionaler Snapshot, aus dem fortgesetzt wird.
        save_path: Optionale Datei, in die der Endzustand gespeichert wird.
        trace_path: Optionale Datei, in die der Lauf als JSONL-Trace mitgeschnitten wird.
    """
    setup_logging()

    console.rule("[bold green]NanoSim-Pet Terrarium[/bold green]")
    console.print(f"Modell: [cyan]{model}[/cyan] | Ticks: [cyan]{num_ticks}[/cyan]")
    if load_path:
        console.print(f"Fortsetzen von: [cyan]{load_path}[/cyan]")
    console.print()

    # Welt + Profile bereitstellen (frisch oder aus Snapshot)
    world, profiles, start_tick = _build_world_and_profiles(load_path)
    router = LlamaRouter(model=model, base_url=base_url)
    bus = EventBus()

    agents: list[BaseAgent] = []
    for profile in profiles:
        agent = BaseAgent(profile=profile, router=router)
        agents.append(agent)
        console.print(
            f"  🐾 [bold]{profile.name}[/bold] ({profile.agent_id}) "
            f"→ {profile.location_id}"
        )

    console.print()

    # EventBus verdrahten
    for agent in agents:
        bus.subscribe(
            agent_id=agent.agent_id,
            handler=agent.receive_event,
            location_fn=lambda a=agent: a.profile.location_id,
        )

    # Optionaler Trace-Recorder
    recorder = None
    if trace_path:
        recorder = TraceWriter(
            trace_path, model=model, agent_ids=[a.agent_id for a in agents],
        )

    # Simulation starten (Tick-Zähler aus dem Snapshot fortsetzen)
    engine = TickEngine(agents=agents, world=world, bus=bus, recorder=recorder)
    engine.tick_count = start_tick

    console.rule("[bold yellow]Simulation startet[/bold yellow]")
    try:
        await engine.run(num_ticks=num_ticks)
    finally:
        if recorder is not None:
            recorder.close()
    console.rule("[bold green]Simulation beendet[/bold green]")

    if trace_path:
        console.print(f"\n[green]Trace gespeichert:[/green] {trace_path}")

    # Endzustand optional speichern
    if save_path:
        save_snapshot(
            save_path, world=world, profiles=profiles, tick_count=engine.tick_count,
        )
        console.print(f"\n[green]Zustand gespeichert:[/green] {save_path}")

    # Zusammenfassung
    console.print()
    console.print("[bold]Endstand:[/bold]")
    for agent in agents:
        p = agent.profile
        s = p.stats
        console.print(
            f"  {p.name} @ {p.location_id} | "
            f"stamina={s.stamina:.2f} mood={s.mood:.2f} hunger={s.hunger:.2f} | "
            f"memory={len(p.memory)} Einträge"
        )


def build_parser() -> argparse.ArgumentParser:
    """Baue den CLI-Argument-Parser. Ausgelagert für Testbarkeit."""
    parser = argparse.ArgumentParser(description="NanoSim-Pet Terrarium")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama-Modellname")
    parser.add_argument("--ticks", type=int, default=5, help="Anzahl Ticks")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument(
        "--load", default=None,
        help="Snapshot-Datei laden und Simulation fortsetzen",
    )
    parser.add_argument(
        "--save", default=None,
        help="Weltzustand nach der Simulation in diese Datei speichern",
    )
    parser.add_argument(
        "--trace", default=None,
        help="Lauf als JSONL-Trace mitschneiden (für Report/Replay)",
    )
    parser.add_argument(
        "--replay", default=None,
        help="Aufgezeichneten Trace abspielen (ohne LLM) statt zu simulieren",
    )
    parser.add_argument(
        "--report", default=None,
        help="Aus dem --replay-Trace einen HTML-Report in diese Datei schreiben",
    )
    return parser


def main() -> None:
    """CLI-Einstiegspunkt."""
    parser = build_parser()
    args = parser.parse_args()

    # Replay-Modus: aufgezeichneten Lauf nutzen, ohne zu simulieren.
    if args.replay:
        if args.report:
            write_report(args.replay, args.report)
            console.print(f"[green]Report geschrieben:[/green] {args.report}")
        else:
            play_trace(args.replay)
        return

    asyncio.run(run_terrarium(
        model=args.model,
        num_ticks=args.ticks,
        base_url=args.url,
        load_path=args.load,
        save_path=args.save,
        trace_path=args.trace,
    ))


if __name__ == "__main__":
    main()
