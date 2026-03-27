# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NanoSim-Pet ist ein lokales Multi-Agenten-Terrarium. LLM-Agenten leben in Räumen, haben Bedürfnisse und interagieren miteinander. Alle LLM-Calls laufen ausschließlich über **Ollama** (AsyncClient, lokal). Python 3.10+, v0.1.0.

## Build & Install

```bash
pip install -e ".[dev]"
```

Voraussetzung: Laufender Ollama-Server auf `localhost:11434`.

## Testing

```bash
pytest                      # alle Tests mit Coverage
pytest -m core              # schnelle Validierung
pytest -m "not llm"         # ohne echte Ollama-Calls
pytest tests/unit/test_foo.py::test_bar  # einzelner Test
```

## Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture

Source in `src/nanosim/`:

- **models.py** — Alle Pydantic-Modelle: `BaseEvent`, `AgentStats`, `AgentProfile`, `AgentAction`, `Room`
- **core/events.py** — `EventBus`: asyncio.Queue-basierter Pub/Sub mit `location_id`-Filterung
- **core/world.py** — `WorldRegistry`: Room-Verwaltung, Agent-Bewegung
- **llm/router.py** — `LlamaRouter`: Ollama-Client mit `asyncio.Semaphore(1)` (VRAM-Schutz), JSON-Extraktion, Pydantic-Validierung, Auto-Retry bei Parse-Fehlern
- **agents/** — Agent-Logik (Persona, Memory)
- **world/** — Room-Definitionen und Terrarium-Layouts

### Kernmechaniken

- **Tick-System**: Diskreter Zeitschritt. Pro Tick: Stats-Decay → LLM-Call pro Agent (sequentiell) → Actions ausführen → Events verteilen
- **Semaphore(1)**: Nur ein Ollama-Call gleichzeitig (VRAM-Schutz für Consumer-GPUs)
- **JSON-Retry**: Bei kaputtem JSON vom LLM → ein automatischer Retry mit Fehlerfeedback an das Modell
- **Memory**: Rolling-List (max 10 Einzeiler, FIFO). Kein LLM-basiertes Summarizing.

## Design-Entscheidungen

Siehe `DESIGN_DOC.md` für die vollständige Architektur-Dokumentation.

## Konventionen

- Kommentare und Docstrings: Deutsch oder Englisch (gemischt, Autor-Präferenz)
- pytest pythonpath ist `src` → Imports: `from nanosim.models import Room`
- Zielmodelle: Llama3-8B, Phi3-mini (kleine lokale Modelle)
