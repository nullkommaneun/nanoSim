"""Gehärtete Orchestrierung der autonomen Experiment-Wellen.

Betriebssicher für unbeaufsichtigten Tagbetrieb:
- Single-Instanz-Lock (flock) — kein Doppelstart.
- Subprozess als eigene Prozessgruppe + Timeout → Gruppen-Kill (keine Zombies).
- VRAM-Preflight (nvidia-smi) → Lauf wird sauber geskippt statt CPU-Spill.
- Resume aus runs.jsonl bis Ziel-n je Config; randomisierte Reihenfolge (Seed geloggt).
- Heartbeat-Datei + Manifest + stderr-Einfang + exception-sicherer Notify-Hook.
- Fehlertolerant: ein Lauf-/Render-Fehler killt nie den Sweep.

Reine Entscheidungslogik (Preflight-Bedarf, Resume-Auswahl, Status-Klassifikation)
ist als testbare Funktionen herausgelöst.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from random import Random

from nanosim.lab.config import (
    MAX_HOURS,
    OLLAMA_URL,
    RESULTS,
    RUN_TIMEOUT,
    TARGET_N,
    active_configs,
    build_cmd,
)
from nanosim.lab.record import RunRecord, append_record, load_records
from nanosim.lab.report import render_report
from nanosim.metrics import compute_metrics
from nanosim.trace import load_trace

REPO = RESULTS.parent.parent
RUNS_FILE = RESULTS / "runs.jsonl"
REPORT_FILE = RESULTS / "report.md"
LOG_FILE = RESULTS / "lab.log"
LOCK_FILE = RESULTS / "lab.lock"
HEARTBEAT_FILE = RESULTS / "heartbeat.json"
MANIFEST_FILE = RESULTS / "manifest.json"
ERRORS_DIR = RESULTS / "errors"

# Grober VRAM-Bedarf je Modell (MiB). Mit OLLAMA_MAX_LOADED_MODELS=1 ist der
# Spitzenbedarf eines Tier-Laufs das GRÖSSERE der beiden Modelle (Swap, nicht Summe).
_MODEL_MIB = {"gemma3:1b": 1500, "llama3.2:3b": 2800, "llama3.1:8b": 5500}
_VRAM_MARGIN = 700

_stop = False


# ---------------------------------------------------------------------------
# Reine, testbare Entscheidungslogik
# ---------------------------------------------------------------------------

def model_vram_need(cfg: dict) -> int:
    """Spitzen-VRAM-Bedarf (MiB) einer Config (Tier = Maximum, nicht Summe)."""
    need = _MODEL_MIB.get(cfg["model"], 3000)
    if cfg.get("dialogue"):
        need = max(need, _MODEL_MIB.get(cfg["dialogue"], 5500))
    return need


def configs_below_target(
    records: list[RunRecord], configs: list[dict], target: int
) -> list[dict]:
    """Configs, die ihr Ziel-n an ERFOLGREICHEN Läufen noch nicht erreicht haben."""
    counts: dict[str, int] = {}
    for r in records:
        if r.ok:
            counts[r.config] = counts.get(r.config, 0) + 1
    return [c for c in configs if counts.get(c["name"], 0) < target]


def classify_status(heartbeat: dict, now: float, run_timeout: float) -> str:
    """Aus einem Heartbeat GRÜN / HÄNGT / TOT ableiten (rein)."""
    pid = heartbeat.get("pid")
    age = now - heartbeat.get("ts", 0)
    if pid is not None and not _pid_alive(pid):
        return f"TOT — Prozess {pid} läuft nicht mehr (letztes Update vor {int(age)}s)."
    if age > run_timeout + 120:
        return f"HÄNGT — seit {int(age)}s kein Lebenszeichen (über Timeout)."
    return (
        f"GRÜN — Runde {heartbeat.get('round', '?')}, "
        f"{heartbeat.get('runs_ok', '?')} Läufe ok, "
        f"Rest-Budget {heartbeat.get('budget_left', '?')}."
    )


# ---------------------------------------------------------------------------
# Infrastruktur-Helfer
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 — Logging darf nie crashen
        pass


def notify(msg: str) -> None:
    """Exception-sicherer Benachrichtigungs-Hook (per Env LAB_NOTIFY_CMD)."""
    log(f"NOTIFY: {msg}")
    cmd = os.environ.get("LAB_NOTIFY_CMD")
    if not cmd:
        return
    try:
        # cmd ist OPERATOR-eigene Konfiguration (eigene Maschine); die Nachricht
        # geht ausschließlich über stdin (input=msg), wird also NIE in den Befehl
        # interpoliert → keine Injektion über den Nachrichteninhalt.
        subprocess.run(cmd, shell=True, input=msg, text=True, timeout=20)  # noqa: S602
    except Exception:  # noqa: BLE001 — eine fehlgeschlagene Notify killt nie den Sweep
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_ollama() -> bool:
    if ollama_up():
        return True
    if not any(h in OLLAMA_URL for h in ("localhost", "127.0.0.1")):
        log(f"FEHLER: entfernter Ollama unter {OLLAMA_URL} nicht erreichbar.")
        return False
    log("Ollama nicht erreichbar — starte 'ollama serve' (VRAM-begrenzt) ...")
    env = {**os.environ, "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NUM_PARALLEL": "1"}
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    except FileNotFoundError:
        log("FEHLER: 'ollama' nicht gefunden.")
        return False
    for _ in range(30):
        time.sleep(1)
        if ollama_up():
            return True
    return False


def gpu_free_mib() -> int | None:
    """Freier VRAM in MiB laut nvidia-smi; None wenn nicht verfügbar (→ kein Skip)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def vram_ok(cfg: dict) -> bool:
    free = gpu_free_mib()
    if free is None:
        return True  # keine GPU-Telemetrie → nicht blockieren
    return free >= model_vram_need(cfg) + _VRAM_MARGIN


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "—"
    except Exception:  # noqa: BLE001
        return "—"


def write_heartbeat(**fields) -> None:
    data = {"pid": os.getpid(), "ts": time.time(), "human_ts": _ts(), **fields}
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        tmp = HEARTBEAT_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(HEARTBEAT_FILE)
    except Exception:  # noqa: BLE001
        pass


def read_status() -> str:
    if not HEARTBEAT_FILE.exists():
        return "Kein Heartbeat — das Labor läuft gerade nicht."
    try:
        hb = json.loads(HEARTBEAT_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "Heartbeat unlesbar."
    return classify_status(hb, time.time(), RUN_TIMEOUT)


def write_report_safe(meta: dict) -> None:
    """Report rendern + schreiben — ein Fehler hier darf den Sweep nie stoppen."""
    try:
        records, skipped = load_records(RUNS_FILE)
        if skipped:
            log(f"WARNUNG: {skipped} kaputte runs.jsonl-Zeile(n) übersprungen.")
        REPORT_FILE.write_text(render_report(records, meta=meta), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log(f"WARNUNG: Report-Rendering fehlgeschlagen: {type(e).__name__}: {e}")


def acquire_lock():
    """Single-Instanz-Lock (flock). Gibt fd zurück oder None, wenn schon belegt."""
    import fcntl

    RESULTS.mkdir(parents=True, exist_ok=True)
    fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd.write(str(os.getpid()))
        fd.flush()
        return fd
    except OSError:
        fd.close()
        return None


def _save_stderr(cfg: dict, idx: int, err: str) -> None:
    if not err:
        return
    try:
        ERRORS_DIR.mkdir(parents=True, exist_ok=True)
        (ERRORS_DIR / f"{cfg['name']}_{idx}.err").write_text(err[-4000:], encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def run_one(cfg: dict, idx: int, seed: int | None) -> RunRecord:
    """Eine Simulation als eigene Prozessgruppe; Ergebnis als RunRecord."""
    trace_path = RESULTS / f"_tmp_{cfg['name']}_{idx}.jsonl"
    cmd = build_cmd(cfg, trace_path, seed=seed)
    t0 = time.time()
    status = "ok"
    metrics: dict = {}
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, err = proc.communicate(timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            err = ""
            status = "timeout"
        else:
            if proc.returncode != 0:
                status = "oom" if "memory" in (err or "").lower() else "error"
                _save_stderr(cfg, idx, err)
            else:
                metrics = compute_metrics(load_trace(trace_path))
    except Exception as e:  # noqa: BLE001
        status = "error"
        log(f"  Ausnahme bei {cfg['name']}: {type(e).__name__}: {e}")
    finally:
        trace_path.unlink(missing_ok=True)

    if status != "ok":
        log(f"  Lauf fehlgeschlagen ({cfg['name']}): {status}")
    return RunRecord(
        round=idx,
        config=cfg["name"],
        ts=time.time(),
        status=status,
        metrics=metrics,
        duration_s=round(time.time() - t0, 1),
        seed=seed,
        model=cfg.get("model"),
        dialogue=cfg.get("dialogue"),
        world=cfg.get("world"),
        prompt_variant=cfg.get("prompt_variant"),
        memory_window=cfg.get("memory_window"),
    )


def _kill_group(proc: subprocess.Popen) -> None:
    """Die ganze Prozessgruppe beenden (SIGTERM → SIGKILL), keine Orphans."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(3)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=10)
    except Exception:  # noqa: BLE001
        pass


def _on_signal(signum, _frame) -> None:
    global _stop
    _stop = True
    log(f"Signal {signum} empfangen — beende nach aktuellem Lauf sauber.")


def _interruptible_sleep(seconds: float) -> None:
    """Schlafen, aber auf _stop reagieren (Stop wird nicht 60s blockiert)."""
    end = time.time() + seconds
    while not _stop:
        remaining = end - time.time()
        if remaining <= 0:
            break
        time.sleep(min(2.0, remaining))


def write_manifest(configs: list[dict], experiment: str) -> None:
    data = {
        "started": _ts(),
        "host": os.uname().nodename,
        "experiment": experiment,
        "commit": git_commit(),
        "ollama_url": OLLAMA_URL,
        "target_n": TARGET_N,
        "configs": [c["name"] for c in configs],
    }
    try:
        RESULTS.mkdir(parents=True, exist_ok=True)
        MANIFEST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock()
    if lock is None:
        print("Labor läuft bereits (Lock belegt) — Abbruch.")
        return

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    experiment = os.environ.get("LAB_EXPERIMENT", "all")
    configs = active_configs(experiment)
    commit = git_commit()
    write_manifest(configs, experiment)
    seed_base = int(os.environ.get("LAB_SEED", "12345"))
    log(
        f"=== Labor gestartet · Experiment '{experiment}' · {len(configs)} Configs · "
        f"Ziel-n {TARGET_N} · Commit {commit} · max {MAX_HOURS}h ==="
    )
    notify(f"nanoSim-Labor gestartet ({experiment}, {len(configs)} Configs).")

    start = time.time()
    rng = Random(seed_base)
    round_i = 0
    runs_ok = 0
    try:
        while not _stop and time.time() - start < MAX_HOURS * 3600:
            records, _ = load_records(RUNS_FILE)
            remaining = configs_below_target(records, configs, TARGET_N)
            if not remaining:
                log("Ziel-n für alle Configs erreicht — fertig.")
                break
            round_i += 1
            order = remaining[:]
            rng.shuffle(order)
            log(f"--- Runde {round_i} · {len(order)} Configs (Seed {seed_base}) ---")
            ran_something = False
            for cfg in order:
                if _stop:
                    break
                if not ensure_ollama():
                    _interruptible_sleep(30)
                    continue
                # VRAM-Preflight: zu wenig frei → NICHT als Ergebnis werten,
                # nur überspringen (Umgebungs-Zustand, kein Experiment-Outcome).
                if not vram_ok(cfg):
                    log(f"  SKIP {cfg['name']}: zu wenig freier VRAM (warte auf Freigabe).")
                    continue
                run_seed = seed_base * 1000 + round_i
                rec = run_one(cfg, round_i, run_seed)
                append_record(RUNS_FILE, rec)
                ran_something = True
                if rec.ok:
                    runs_ok += 1
                budget_left = int(MAX_HOURS * 3600 - (time.time() - start))
                write_heartbeat(
                    round=round_i, runs_ok=runs_ok, current_config=cfg["name"],
                    budget_left=f"{budget_left // 3600}h{(budget_left % 3600) // 60}m",
                    ollama_up=True, experiment=experiment,
                )
                if rec.ok:
                    log(
                        f"  {cfg['name']}: reaction_chains={rec.metrics.get('reaction_chains')} "
                        f"({rec.duration_s}s)"
                    )
            write_report_safe(_report_meta(experiment, commit, start))
            # Konnte nichts laufen (VRAM belegt / Ollama weg) → mit Backoff warten,
            # statt in einer Leerlaufschleife die CPU zu verbrennen.
            if not ran_something and not _stop:
                log("Runde komplett übersprungen (VRAM/Ollama) — warte 60s auf Ressourcen.")
                write_heartbeat(
                    round=round_i, runs_ok=runs_ok,
                    current_config="(wartet auf VRAM)", waiting=True,
                )
                _interruptible_sleep(60)
            else:
                log(f"Runde {round_i} fertig, Report aktualisiert.")
    finally:
        write_report_safe(_report_meta(experiment, commit, start))
        write_heartbeat(round=round_i, runs_ok=runs_ok, done=True)
        msg = f"nanoSim-Labor beendet · {runs_ok} erfolgreiche Läufe · Experiment {experiment}."
        log(f"=== {msg} ===")
        notify(msg)
        lock.close()


def _report_meta(experiment: str, commit: str, start: float) -> dict:
    budget_left = int(MAX_HOURS * 3600 - (time.time() - start))
    return {
        "timestamp": _ts(),
        "experiment": experiment,
        "commit": commit,
        "health": read_status(),
        "budget_left": f"{max(0, budget_left) // 3600}h{(max(0, budget_left) % 3600) // 60}m",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(read_status())
    else:
        main()
