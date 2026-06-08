"""Tests für den Replay/Playback eines aufgezeichneten Laufs."""

from nanosim.models import ActionType, AgentStats, BaseEvent, EventType
from nanosim.replay import format_event, render_trace_lines, summarize_tick
from nanosim.trace import (
    AgentSnapshot,
    Decision,
    TickRecord,
    Trace,
    TraceMeta,
)


def _snap(hunger=0.3):
    return AgentSnapshot(
        agent_id="cat_01", name="Whiskers", location_id="kitchen",
        stats=AgentStats(stamina=0.8, mood=0.7, hunger=hunger),
    )


class TestFormatEvent:
    def test_speak(self):
        ev = BaseEvent(
            type=EventType.AGENT_SPEAK, source="cat_01",
            payload={"message": "Miau!"},
        )
        s = format_event(ev)
        assert "cat_01" in s
        assert "Miau!" in s

    def test_move(self):
        ev = BaseEvent(
            type=EventType.AGENT_MOVE, source="dog_01",
            payload={"from": "garden", "to": "kitchen"},
        )
        s = format_event(ev)
        assert "dog_01" in s
        assert "kitchen" in s

    def test_use(self):
        ev = BaseEvent(
            type=EventType.AGENT_USE, source="cat_01",
            payload={"object": "futternapf"},
        )
        s = format_event(ev)
        assert "futternapf" in s

    def test_rest(self):
        ev = BaseEvent(type=EventType.AGENT_REST, source="cat_01")
        s = format_event(ev)
        assert "cat_01" in s


class TestSummarizeTick:
    def test_header_contains_tick_number(self):
        tick = TickRecord(tick=7, agents=[_snap()], decisions=[], events=[])
        lines = summarize_tick(tick)
        assert any("7" in ln for ln in lines)

    def test_renders_speak_event(self):
        ev = BaseEvent(
            type=EventType.AGENT_SPEAK, source="cat_01",
            payload={"message": "Hallo Welt"},
        )
        tick = TickRecord(
            tick=0, agents=[_snap()],
            decisions=[Decision(agent_id="cat_01", action=ActionType.SPEAK, message="Hallo Welt")],
            events=[ev],
        )
        lines = summarize_tick(tick)
        assert any("Hallo Welt" in ln for ln in lines)

    def test_quiet_tick_still_has_header(self):
        tick = TickRecord(tick=2, agents=[_snap()], decisions=[], events=[])
        lines = summarize_tick(tick)
        assert len(lines) >= 1


class TestCliReplayBranch:
    def test_replay_plays_and_skips_simulation(self, monkeypatch):
        """Mit --replay wird abgespielt und KEINE Simulation (Ollama) gestartet."""
        import sys

        import nanosim.main as m

        called = {}
        monkeypatch.setattr(m, "play_trace", lambda p: called.__setitem__("play", p))
        monkeypatch.setattr(m.asyncio, "run", lambda coro: called.__setitem__("ran", True))
        monkeypatch.setattr(sys, "argv", ["nanosim", "--replay", "x.jsonl"])

        m.main()

        assert called.get("play") == "x.jsonl"
        assert "ran" not in called

    def test_report_writes_html_and_skips_simulation(self, monkeypatch):
        """--replay + --report erzeugt HTML statt Konsolen-Wiedergabe, ohne Ollama."""
        import sys

        import nanosim.main as m

        called = {}
        monkeypatch.setattr(m, "write_report", lambda src, out: called.__setitem__("report", (src, out)))
        monkeypatch.setattr(m, "play_trace", lambda p: called.__setitem__("play", p))
        monkeypatch.setattr(m.asyncio, "run", lambda coro: called.__setitem__("ran", True))
        monkeypatch.setattr(sys, "argv", ["nanosim", "--replay", "x.jsonl", "--report", "out.html"])

        m.main()

        assert called.get("report") == ("x.jsonl", "out.html")
        assert "play" not in called
        assert "ran" not in called


class TestRenderTrace:
    def test_renders_all_ticks_in_order(self):
        trace = Trace(
            meta=TraceMeta(model="m", agent_ids=["cat_01"]),
            ticks=[
                TickRecord(tick=0, agents=[_snap()], decisions=[], events=[]),
                TickRecord(tick=1, agents=[_snap()], decisions=[], events=[]),
            ],
        )
        text = "\n".join(render_trace_lines(trace))
        assert "0" in text and "1" in text
        # Tick 0 muss vor Tick 1 erscheinen
        assert text.index("TICK 0") < text.index("TICK 1")
