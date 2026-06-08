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
- **llm/router.py** — `LlamaRouter`: Ollama-Client mit `asyncio.Semaphore(1)` (VRAM-Schutz), Structured Outputs (`format=schema` erzwingt JSON), Pydantic-Validierung, Auto-Retry als Notnagel
- **agents/** — Agent-Logik (Persona, Memory)
- **world/** — Room-Definitionen und Terrarium-Layouts
- **persistence.py** — `Snapshot`-Modell + `save_snapshot`/`load_snapshot`/`world_from_snapshot`: Weltzustand als JSON speichern/laden (CLI: `--save`/`--load`)
- **trace.py** — `TraceWriter`/`load_trace`: Lauf als JSONL mitschneiden (TickRecord pro Tick, inkl. echter LLM-Entscheidungen). Feldnamen an OpenTelemetry-GenAI angelehnt (CLI: `--trace`)
- **replay.py** — Aufgezeichneten Lauf ohne LLM abspielen (faithful by construction). `format_event`/`summarize_tick`/`render_trace_lines` + `play_trace` (CLI: `--replay`)
- **report.py** — Eigenständiger HTML-Report aus einem Trace (Plotly, offline): Lebenskurven, Reaktionsketten (Causality-Baum), Raumwechsel, Tick-Protokoll. Optionale Abhängigkeit `[report]` (CLI: `--replay X --report Y.html`)

### Kernmechaniken

- **Tick-System**: Diskreter Zeitschritt. Pro Tick: Stats-Decay → LLM-Call pro Agent (sequentiell) → Actions ausführen → Events verteilen
- **Semaphore(1)**: Nur ein Ollama-Call gleichzeitig (VRAM-Schutz für Consumer-GPUs)
- **Structured Outputs**: Das JSON-Schema wird als `format` an Ollama übergeben → strukturell erzwungenes JSON. Temperatur bleibt beim Modell-Default (Verhaltens-Vielfalt erwünscht, kein `temperature=0`).
- **JSON-Retry**: Notnagel, falls trotz `format` etwas schiefgeht → ein automatischer Retry mit Fehlerfeedback an das Modell
- **Memory**: Rolling-List (max 10 Einzeiler, FIFO). Kein LLM-basiertes Summarizing.
- **Causality-Kappung**: `BaseEvent.causality_depth` begrenzt Reaktions-Ketten. Antwort auf Tiefe `d` → eigene Tiefe `d+1`; über `MAX_CAUSALITY_DEPTH` (5) schweigt der Agent. Verhindert Endlos-Echo zwischen Agenten.

## Design-Entscheidungen

Siehe `DESIGN_DOC.md` für die vollständige Architektur-Dokumentation.

## Konventionen

- Kommentare und Docstrings: Deutsch oder Englisch (gemischt, Autor-Präferenz)
- pytest pythonpath ist `src` → Imports: `from nanosim.models import Room`
- Zielmodelle: Llama3-8B, Phi3-mini (kleine lokale Modelle)
