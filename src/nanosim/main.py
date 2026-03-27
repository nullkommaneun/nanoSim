"""NanoSim-Pet Einstiegspunkt — Startet das Terrarium."""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.logging import RichHandler

from nanosim.agents.base import BaseAgent
from nanosim.core.events import EventBus
from nanosim.core.tick import TickEngine
from nanosim.llm.router import LlamaRouter
from nanosim.world.personas import create_default_agents
from nanosim.world.rooms import create_default_world

console = Console()


def setup_logging(level: int = logging.INFO) -> None:
    """Konfiguriere Rich-Logging für hübsche Konsolenausgabe."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


async def run_terrarium(
    model: str = "llama3.1:8b",
    num_ticks: int = 5,
    base_url: str = "http://localhost:11434",
) -> None:
    """Starte eine NanoSim-Pet Simulation.

    Args:
        model: Ollama-Modellname.
        num_ticks: Anzahl Ticks die simuliert werden.
        base_url: Ollama-Server URL.
    """
    setup_logging()
    logger = logging.getLogger("nanosim")

    console.rule("[bold green]NanoSim-Pet Terrarium[/bold green]")
    console.print(f"Modell: [cyan]{model}[/cyan] | Ticks: [cyan]{num_ticks}[/cyan]")
    console.print()

    # Welt aufbauen
    world = create_default_world()
    router = LlamaRouter(model=model, base_url=base_url)
    bus = EventBus()

    # Agenten erstellen und in der Welt platzieren
    profiles = create_default_agents()
    agents: list[BaseAgent] = []

    for profile in profiles:
        agent = BaseAgent(profile=profile, router=router)
        agents.append(agent)
        world.get_room(profile.location_id).occupants.add(profile.agent_id)
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

    # Simulation starten
    engine = TickEngine(agents=agents, world=world, bus=bus)

    console.rule("[bold yellow]Simulation startet[/bold yellow]")
    await engine.run(num_ticks=num_ticks)
    console.rule("[bold green]Simulation beendet[/bold green]")

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


def main() -> None:
    """CLI-Einstiegspunkt."""
    import argparse

    parser = argparse.ArgumentParser(description="NanoSim-Pet Terrarium")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama-Modellname")
    parser.add_argument("--ticks", type=int, default=5, help="Anzahl Ticks")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    args = parser.parse_args()

    asyncio.run(run_terrarium(
        model=args.model,
        num_ticks=args.ticks,
        base_url=args.url,
    ))


if __name__ == "__main__":
    main()
