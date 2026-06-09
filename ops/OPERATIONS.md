# Betrieb der autonomen Experiment-Wellen

Diese Anleitung richtet das gehärtete Labor für **unbeaufsichtigten Tagbetrieb** ein.
Alles läuft lokal auf der RTX 3070; der Code ist betriebssicher (Lock, Resume,
Timeouts, VRAM-Preflight, Heartbeat). Die folgenden Schritte machst **du einmalig**
am Betriebssystem.

---

## 1. Einmalig: Ollama VRAM-sicher konfigurieren (wichtig)

Standardmäßig kann Ollama mehrere Modelle gleichzeitig in den 8 GB VRAM laden →
OOM/CPU-Spill. Begrenze das auf **ein Modell**:

```bash
sudo systemctl edit ollama.service
```

Im Editor einfügen, speichern:

```ini
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
```

Dann übernehmen und prüfen:

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama ps            # während eines Laufs: nie mehr als 1 Modell resident
```

---

## 2. Labor als Dienst installieren (Auto-Restart)

```bash
mkdir -p ~/.config/systemd/user
cp ops/nanosim-lab.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

Optional, damit der Dienst auch ohne aktive Login-Sitzung weiterläuft:

```bash
sudo loginctl enable-linger "$USER"
```

---

## 3. Steuern

```bash
# Welle starten (läuft, bis Ziel-n je Config erreicht oder 10h-Deckel)
systemctl --user start nanosim-lab

# Läuft es? Gesundheit aus der Ferne:
systemctl --user status nanosim-lab
.venv/bin/python experiments/lab.py status     # GRÜN / HÄNGT / TOT

# Sauber stoppen (beendet nach dem aktuellen Lauf):
systemctl --user stop nanosim-lab
```

Welches Experiment? In `~/.config/systemd/user/nanosim-lab.service` die Zeile
`Environment=LAB_EXPERIMENT=...` setzen (`all`, `models`, `prompt`, `world`,
`memory`), dann `systemctl --user daemon-reload && systemctl --user restart nanosim-lab`.

Ohne systemd geht es auch direkt:

```bash
LAB_EXPERIMENT=all nohup .venv/bin/python experiments/lab.py \
    > experiments/results/nohup.out 2>&1 &
```

---

## 4. Benachrichtigung (optional, empfohlen)

Damit du **während der Arbeit** von einem Crash/Ende erfährst, setze einen
Notify-Befehl. Die Nachricht kommt über stdin:

```ini
# in der Service-Datei, Abschnitt [Service]:
Environment=LAB_NOTIFY_CMD=mail -s "nanoSim-Labor" thom.ehr@gmail.com
```

Oder per Handy-Push über ntfy.sh:

```ini
Environment=LAB_NOTIFY_CMD=curl -s -d @- ntfy.sh/DEIN-GEHEIMES-THEMA
```

---

## 5. Ergebnisse ansehen

```bash
cat experiments/results/report.md          # Ranking mit 95%-Konfidenzintervallen
cat experiments/results/heartbeat.json     # aktueller Zustand
ls  experiments/results/errors/            # eingefangene Fehler (falls vorhanden)
```

Der Report nennt bei überlappenden Konfidenzintervallen bewusst **keinen**
Einzel-Sieger, sondern eine „Top-Gruppe gleichauf", und nimmt degenerierte
Configs (keine Interaktion) aus dem Ranking — ehrliche Statistik statt
Rausch-Sieger.

---

## Was die Härtung garantiert (Definition of Done)

- Übersteht Crash/Reboot: Auto-Restart, Resume aus `runs.jsonl` (beginnt nicht bei 0).
- Kann nicht doppelt laufen (flock-Lock).
- OOM-sicher: `OLLAMA_MAX_LOADED_MODELS=1` + VRAM-Preflight (Lauf wird geskippt statt CPU-Spill).
- Aus der Ferne diagnostizierbar: `status` + eine Push/Mail bei Crash/Ende.
- Keine Leichen: temp-Traces werden aufgeräumt; eine kaputte Zeile blockiert nichts.
- Ehrliche Statistik: Konfidenzintervalle, Erfolgsquote, degenerierte Configs geflaggt.
- Reproduzierbar genug: Seed pro Lauf protokolliert; Reihenfolge randomisiert.
- Unit-getestet: die Orchestrierungs-Logik liegt im Paket `nanosim.lab`.
