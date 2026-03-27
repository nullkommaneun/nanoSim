# NanoSim-Pet

Ein lokales, verspieltes Multi-Agenten-Terrarium. KI-gesteuerte Haustiere leben in Räumen, haben Bedürfnisse, bewegen sich frei und reden miteinander — komplett offline, auf deiner eigenen Hardware, via [Ollama](https://ollama.com/).

## Was passiert hier?

NanoSim-Pet simuliert eine kleine Welt mit Räumen (Küche, Garten, Wohnzimmer, Balkon) und KI-Agenten (Katze, Hund, Papagei). Jeder Agent hat:

- **Bedürfnisse** (Energie, Stimmung, Hunger) die über Zeit verfallen
- **Erinnerungen** an vergangene Interaktionen
- **Eigenständige Entscheidungen** via LLM (sprechen, bewegen, Objekte benutzen, ruhen)

Pro Tick entscheidet jeder Agent selbstständig, was er tut. Agenten im selben Raum hören einander und reagieren aufeinander.

```
Tick 0 | Whiskers (Katze, Küche)  → speak: "Wo ist meine Milch?"
Tick 0 | Bello (Hund, Garten)     → move south → Küche
Tick 1 | Bello (Hund, Küche)      → speak: "Hallo! Ich suche Fressen!"
Tick 2 | Whiskers                  → speak: "Hallo Bello! Hast du meine Milch gesehen?"
```

---

## Voraussetzungen

- **Python 3.10+**
- **Ollama** mit mindestens einem Modell (empfohlen: `llama3.1:8b`)
- **Git**

### Ollama installieren

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows: Download von https://ollama.com/download
```

Ollama-Server starten und ein Modell ziehen:

```bash
# Server starten (läuft im Hintergrund auf Port 11434)
ollama serve

# In einem neuen Terminal: Modell herunterladen (~4.7 GB)
ollama pull llama3.1:8b
```

Prüfen ob Ollama läuft:

```bash
curl http://localhost:11434/api/tags
```

---

## Installation (isoliert mit venv)

Das Projekt wird in einer virtuellen Python-Umgebung installiert, vollständig isoliert vom System-Python. Kein `sudo`, kein globales pip.

### 1. Repository klonen

```bash
git clone https://github.com/nullkommaneun/nanoSim.git
cd nanoSim
```

### 2. Virtuelle Umgebung erstellen

```bash
python3 -m venv .venv
```

Das erstellt einen Ordner `.venv/` im Projektverzeichnis mit einer eigenen Python-Installation.

### 3. Virtuelle Umgebung aktivieren

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

Nach der Aktivierung steht `(.venv)` vor deinem Prompt. Alle folgenden Befehle laufen jetzt im venv.

### 4. NanoSim-Pet installieren

```bash
# Nur zum Ausführen
pip install -e .

# Mit Entwickler-Tools (Tests, Linter)
pip install -e ".[dev]"
```

`-e` steht für "editable" — Änderungen am Code wirken sofort, ohne Neuinstallation.

### 5. Installation prüfen

```bash
nanosim --help
```

Erwartete Ausgabe:

```
usage: nanosim [-h] [--model MODEL] [--ticks TICKS] [--url URL]

NanoSim-Pet Terrarium

options:
  -h, --help     show this help message and exit
  --model MODEL  Ollama-Modellname
  --ticks TICKS  Anzahl Ticks
  --url URL      Ollama URL
```

### Deaktivieren

```bash
deactivate
```

---

## Starten (Headless)

NanoSim-Pet läuft komplett im Terminal — kein Browser, kein GUI, kein Server. Die Ausgabe erfolgt über Rich-formatiertes Logging direkt in die Konsole.

### Standard-Simulation (5 Ticks, 3 Agenten)

```bash
nanosim
```

### Modell wählen

```bash
# Llama 3.1 8B (Standard, empfohlen)
nanosim --model llama3.1:8b

# Kleineres Modell (schneller, weniger VRAM)
nanosim --model gemma3:1b

# Größeres Modell (besser, mehr VRAM nötig)
nanosim --model llama3.1:latest
```

### Anzahl Ticks festlegen

```bash
# Kurzer Test: 2 Ticks
nanosim --ticks 2

# Längere Simulation: 20 Ticks
nanosim --ticks 20
```

### Ollama auf anderem Host/Port

```bash
nanosim --url http://192.168.1.100:11434
```

### Alles kombiniert

```bash
nanosim --model gemma3:4b --ticks 10 --url http://localhost:11434
```

### Als Python-Modul starten

```bash
python -m nanosim.main --model llama3.1:8b --ticks 3
```

---

## Testen

Alle Tests laufen ohne laufenden Ollama-Server (außer Integrationstests). Der LLM-Router wird in Unit-Tests komplett gemockt.

### Alle Unit-Tests

```bash
pytest tests/unit/ -v
```

### Nur schnelle Core-Tests

```bash
pytest -m core -v
```

### Tests ohne echte LLM-Calls (kein Ollama nötig)

```bash
pytest -m "not llm" -v
```

### Ein einzelner Test

```bash
pytest tests/unit/test_models.py::TestAgentStats::test_defaults -v
```

### Eine einzelne Test-Datei

```bash
pytest tests/unit/test_agent.py -v
```

### Integrationstest (braucht laufendes Ollama)

```bash
pytest tests/integration/ -v -s
```

Der Integrationstest führt 3 echte Ticks mit 2 Agenten und echtem Ollama durch.

### Mit Coverage-Report

```bash
# HTML-Report (wird in htmlcov/ geschrieben)
pytest --cov=nanosim --cov-report=html

# Terminal-Report
pytest --cov=nanosim --cov-report=term-missing
```

### Tests mit sichtbarem Logging (Debugging)

```bash
pytest tests/integration/ -v -s --log-cli-level=INFO
```

Zeigt alle Prompts, LLM-Antworten und Actions live in der Konsole.

### Tests mit Debug-Level Logging

```bash
pytest tests/integration/ -v -s --log-cli-level=DEBUG
```

---

## Debugging

### Verbose-Logging im Code aktivieren

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Einzelnen Agent isoliert testen

```python
import asyncio
from nanosim.agents.base import BaseAgent
from nanosim.llm.router import LlamaRouter
from nanosim.world.rooms import create_simple_world
from nanosim.world.personas import create_cat

async def test_single_agent():
    router = LlamaRouter(model="llama3.1:8b")
    world = create_simple_world()
    profile = create_cat()
    world.get_room(profile.location_id).occupants.add(profile.agent_id)

    agent = BaseAgent(profile=profile, router=router)
    event = await agent.tick(world, tick=0)
    print(f"Action: {event}")
    print(f"Memory: {agent.profile.memory}")

asyncio.run(test_single_agent())
```

### LLM-Router isoliert testen

```python
import asyncio
from nanosim.llm.router import LlamaRouter
from nanosim.models import AgentAction

async def test_router():
    router = LlamaRouter(model="llama3.1:8b")
    result = await router.think(
        prompt="Du bist eine Katze in der Küche. Was tust du?",
        response_model=AgentAction,
        system="Antworte als JSON.",
    )
    print(f"Parsed: {result}")

asyncio.run(test_router())
```

### JSON-Parsing debuggen

```python
from nanosim.llm.router import LlamaRouter

# Simuliere kaputte LLM-Antwort
raw = '```json\n{"action": "speak", "message": "Miau!"}\n```'
cleaned = LlamaRouter._extract_json(raw)
print(f"Extracted: {cleaned}")

from nanosim.models import AgentAction
result = LlamaRouter._parse_and_validate(raw, AgentAction)
print(f"Validated: {result}")
```

---

## Linting

```bash
# Code prüfen
ruff check src/ tests/

# Code automatisch formatieren
ruff format src/ tests/
```

---

## Architektur

```
TickEngine          ← Treibt die Weltzeit (Tick für Tick)
    │
    ├── decay_stats()     Stats-Verfall pro Tick
    │
    ├── BaseAgent.tick()  Jeder Agent denkt + handelt
    │       │
    │       ├── build_prompt()    Situations-Kontext bauen
    │       ├── LlamaRouter.think()   LLM-Call via Ollama
    │       └── _execute()        Action ausführen
    │
    └── EventBus.drain()  Events an Agenten im selben Room verteilen

WorldRegistry       ← Verwaltet Rooms und Agent-Positionen
LlamaRouter         ← Semaphore(1) schützt VRAM, Auto-Retry bei kaputtem JSON
```

### Warum Semaphore(1)?

Consumer-GPUs (8-16 GB VRAM) können nur einen Inference-Request gleichzeitig bedienen. Parallele Requests führen zu OOM oder extremem Thrashing. Der Semaphore stellt sicher, dass immer nur ein Agent gleichzeitig "denkt".

### Warum JSON-Retry?

Kleine Modelle (Llama3-8B, Phi3-mini, Gemma) produzieren regelmäßig kaputtes JSON. Bei einem Parse-Fehler bekommt das Modell die Fehlermeldung zurück und einen zweiten Versuch. Schlägt auch das fehl, führt der Agent `idle` aus.

---

## Projektstruktur

```
src/nanosim/
├── models.py           Pydantic-Modelle (BaseEvent, AgentStats, Room, ...)
├── main.py             CLI-Einstiegspunkt
├── core/
│   ├── events.py       EventBus (asyncio.Queue + Room-Filterung)
│   ├── world.py        WorldRegistry
│   └── tick.py         TickEngine + decay_stats
├── agents/
│   ├── base.py         BaseAgent (Inbox, Tick, Action-Execution)
│   └── prompt.py       Prompt-Builder
├── llm/
│   └── router.py       LlamaRouter (Ollama, Semaphore, JSON-Repair)
└── world/
    ├── rooms.py        Room-Presets + Terrarium-Layouts
    └── personas.py     Agent-Personas (Katze, Hund, Papagei)

tests/
├── unit/               84 Tests (kein Ollama nötig)
└── integration/        1 E2E-Test (braucht Ollama)
```

---

## Lizenz

MIT
