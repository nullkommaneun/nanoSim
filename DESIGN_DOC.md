# NanoSim-Pet — Design Document

> Lokales Multi-Agenten-Terrarium. Alle LLM-Calls laufen über Ollama (AsyncClient).
> Zielmodelle: Llama3 8B, Phi3-mini — alles muss mit kaputtem JSON klarkommen.

---

## 1. World & Spatial Engine

### Konzept
Die Welt besteht aus **Rooms**. Jeder Room hat eine `room_id`, eine Liste von `occupants` (Agent-IDs) und eine Liste von `objects` (interagierbare Items). Rooms können über `exits` miteinander verbunden sein.

### Room-Modell
```python
class Room(BaseModel):
    room_id: str                    # z.B. "kitchen", "garden"
    name: str                       # Anzeigename
    description: str                # Für den LLM-Prompt-Kontext
    occupants: set[str] = set()     # Agent-IDs die sich hier befinden
    objects: list[str] = []         # Interagierbare Objekte
    exits: dict[str, str] = {}     # {"north": "garden", "east": "bedroom"}
```

### WorldRegistry
Eine zentrale Registry (`WorldRegistry`) hält alle Rooms und bietet:
- `get_room(room_id) → Room`
- `move_agent(agent_id, from_room, to_room)` — aktualisiert `occupants` beider Rooms
- `agents_in_room(room_id) → set[str]` — für Event-Filterung

### Event-Bus & Location-Filtering
Der Event-Bus ist ein einfacher `asyncio.Queue`-basierter Pub/Sub:

```
Event erzeugt → Bus.publish(event)
                    ↓
              location_id gesetzt?
              ├── JA  → nur Subscriber die in diesem Room sind erhalten das Event
              └── NEIN → Broadcast an alle Subscriber (System-Events)
```

**Entscheidung:** Kein komplexes Topic-System. Filterung ausschließlich über `location_id`. Das reicht für ein Terrarium und hält die Komplexität minimal.

---

## 2. Agent-State-Machine

### AgentStats
Jeder Agent hat Bedürfnisse die über Zeit sinken (Tick-System, siehe §4):

```python
class AgentStats(BaseModel):
    stamina: float = Field(default=1.0, ge=0.0, le=1.0)   # Energie
    mood: float = Field(default=1.0, ge=0.0, le=1.0)      # Stimmung
    hunger: float = Field(default=0.0, ge=0.0, le=1.0)    # 0=satt, 1=verhungert
```

Alle Werte sind normalisiert auf `[0.0, 1.0]`. Das vereinfacht die Prompt-Generierung: wir können dem LLM direkt sagen "Dein Hunger ist bei 0.7/1.0".

### AgentProfile
```python
class AgentProfile(BaseModel):
    agent_id: str
    name: str
    persona: str              # System-Prompt-Fragment, z.B. "Du bist eine neugierige Katze"
    location_id: str          # Aktueller Room
    stats: AgentStats
    inventory: list[str] = []
    memory: list[str] = []    # Komprimierte Erinnerungen (max. 10 Einträge)
```

### Memory-Kompression
Kleine Modelle haben winzige Context-Windows. Strategie:
1. Memory ist eine **Rolling-List** mit max. 10 Einträgen
2. Jeder Eintrag ist ein Einzeiler: `"Tick 42: Habe im Garten eine Blume gefunden"`
3. Wenn die Liste voll ist, wird der **älteste Eintrag** entfernt (FIFO)
4. **Kein LLM-basiertes Summarizing** — das wäre zu teuer für kleine Modelle. Stattdessen: feste Einzeiler-Kompression bei der Erzeugung.

### State-Transitions
Agenten haben keinen expliziten State-Machine-Graphen. Stattdessen entscheidet das LLM pro Tick, welche **Action** der Agent ausführt. Mögliche Actions:

| Action     | Effekt                                          |
|------------|--------------------------------------------------|
| `speak`    | Event an alle Agenten im selben Room             |
| `move`     | Agent wechselt den Room (via WorldRegistry)       |
| `use`      | Interaktion mit einem Objekt                      |
| `rest`     | stamina += 0.2, hunger += 0.05                    |
| `idle`     | Nichts passiert (kostengünstig — kein Follow-up)  |

---

## 3. Ollama-Bottleneck (VRAM-Türsteher)

### Problem
Ein lokaler Ollama-Server kann nur **einen Inference-Request gleichzeitig** effizient bedienen (besonders bei Consumer-GPUs mit 8-16GB VRAM). Parallele Requests führen zu OOM oder massivem Thrashing.

### Lösung: `asyncio.Semaphore(1)`
```
Agent A will denken ──→ acquire() ──→ Ollama-Call ──→ release()
Agent B will denken ──→ wartet...  ──→ acquire()  ──→ Ollama-Call ──→ release()
Agent C will denken ──→ wartet...     wartet...   ──→ acquire() ...
```

### LlamaRouter-Klasse
```python
class LlamaRouter:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self._client = ollama.AsyncClient(host=base_url)
        self._semaphore = asyncio.Semaphore(1)
        self.model = model

    async def think(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        """Sende Prompt an Ollama, parse JSON, validiere gegen Pydantic-Modell."""
```

### JSON-Reparatur-Mechanismus
Kleine Modelle (Llama3-8B, Phi3-mini) produzieren regelmäßig kaputtes JSON:
- Fehlende schließende Klammern
- Trailing Kommas
- Unescapte Quotes in Strings

**Strategie (2 Versuche):**
1. **Versuch 1:** Response parsen → `json.loads()` → `response_model.model_validate()`
2. **Bei Fehler — Versuch 2:** Neuer Prompt mit der Original-Frage + dem Fehlertext:
   ```
   "Deine letzte Antwort war kein valides JSON. Fehler: {error}.
    Antworte NUR mit validem JSON im Format: {schema}"
   ```
3. **Bei erneutem Fehler:** `None` zurückgeben (Agent führt `idle` aus)

**Kein `json_repair`-Library.** Wir halten die Dependency-Liste schlank und nutzen stattdessen das Retry-mit-Fehler-Feedback-Pattern, das dem Modell die Chance gibt, sich selbst zu korrigieren.

### Prompt-Format
Alle Prompts enden mit einer expliziten JSON-Anweisung:
```
Respond ONLY with valid JSON matching this schema:
{response_model.model_json_schema()}

Do not include any text before or after the JSON.
```

---

## 4. Das Tick-System

### Konzept
Ein "Tick" ist ein diskreter Zeitschritt. Pro Tick passiert:

```
┌─────────────────────────────────────────┐
│ TICK N                                   │
│                                          │
│  1. Stats-Decay für alle Agenten         │
│     - hunger += 0.05                     │
│     - stamina -= 0.03                    │
│     - mood: abhängig von hunger/stamina  │
│                                          │
│  2. Für jeden Agenten (sequentiell!):    │
│     a) Kontext bauen (Room, Stats, etc.) │
│     b) LLM-Call via LlamaRouter.think()  │
│     c) Action ausführen                  │
│     d) Event auf den Bus publishen       │
│                                          │
│  3. Event-Bus verarbeiten                │
│     - Zugestellte Events → Agent-Memory  │
│                                          │
│  4. Tick-Counter erhöhen                 │
└─────────────────────────────────────────┘
```

### Warum sequentiell?
Weil der Semaphore(1) eh nur einen LLM-Call gleichzeitig zulässt. Sequentielle Verarbeitung macht die Reihenfolge deterministisch und debugbar. Die Reihenfolge der Agenten wird pro Tick **zufällig gemischt** (Fairness).

### Tick-Intervall
Konfigurierbar, Default: **kein Timer**. Ein Tick wird explizit getriggert (für Entwicklung/Debugging). Später optional: `asyncio.sleep(interval)` zwischen Ticks für Echtzeit-Modus.

### Stats-Decay-Formeln
```python
def decay_stats(stats: AgentStats) -> AgentStats:
    return stats.model_copy(update={
        "hunger": min(1.0, stats.hunger + 0.05),
        "stamina": max(0.0, stats.stamina - 0.03),
        "mood": max(0.0, stats.mood - 0.02 * stats.hunger),  # Hunger drückt Stimmung
    })
```

### Tick-Loop
```python
class TickEngine:
    async def run(self, num_ticks: int | None = None):
        """Laufe num_ticks Ticks, oder endlos wenn None."""
        tick = 0
        while num_ticks is None or tick < num_ticks:
            await self.step()  # Ein einzelner Tick
            tick += 1
```

---

## 5. Zusammenspiel der Komponenten

```
                    ┌──────────────┐
                    │  TickEngine  │
                    └──────┬───────┘
                           │ step()
                           ▼
                ┌──────────────────────┐
                │  WorldRegistry       │
                │  (Rooms, Agents)     │
                └──────────┬───────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ Agent A │ │ Agent B │ │ Agent C │
         └────┬────┘ └────┬────┘ └────┬────┘
              │            │            │
              └────────────┼────────────┘
                           │ think()
                           ▼
                    ┌──────────────┐
                    │ LlamaRouter  │──→ Ollama (localhost:11434)
                    │ Semaphore(1) │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Event-Bus   │
                    │  (asyncio.Q) │
                    └──────────────┘
```

---

## 6. Nicht-Ziele (bewusst ausgelassen)

- **Kein Web-UI** — Terminal-Output via Rich reicht für v0.1
- **Kein Persistence-Layer** — kein SQLite, kein Speichern auf Disk
- **Kein Multi-Model-Routing** — ein Modell für alle Agenten
- **Keine Tool-Use/Function-Calling** — Actions kommen als JSON aus dem LLM
- **Kein verteiltes System** — ein Prozess, ein Event-Loop
