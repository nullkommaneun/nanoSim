"""Tests für den HTML-Report (Plotly, eigenständig/offline)."""

import pytest

from nanosim.models import ActionType, AgentStats, BaseEvent, EventType
from nanosim.report import build_report_html, write_report
from nanosim.trace import (
    AgentSnapshot,
    Decision,
    TickRecord,
    Trace,
    TraceMeta,
    TraceWriter,
)

pytest.importorskip("plotly")


def _snap(agent_id, name, room, hunger):
    return AgentSnapshot(
        agent_id=agent_id, name=name, location_id=room,
        stats=AgentStats(stamina=0.9, mood=0.8, hunger=hunger),
    )


@pytest.fixture
def trace():
    return Trace(
        meta=TraceMeta(model="llama3.1:8b", agent_ids=["cat_01", "dog_01"]),
        ticks=[
            TickRecord(
                tick=0,
                agents=[_snap("cat_01", "Whiskers", "kitchen", 0.05),
                        _snap("dog_01", "Bello", "garden", 0.05)],
                decisions=[Decision(agent_id="cat_01", action=ActionType.SPEAK, message="Wo ist meine Milch?")],
                events=[BaseEvent(id="e1", type=EventType.AGENT_SPEAK, source="cat_01",
                                  location_id="kitchen", payload={"message": "Wo ist meine Milch?"})],
            ),
            TickRecord(
                tick=1,
                agents=[_snap("cat_01", "Whiskers", "kitchen", 0.10),
                        _snap("dog_01", "Bello", "kitchen", 0.10)],
                decisions=[Decision(agent_id="dog_01", action=ActionType.SPEAK, message="Wuff!")],
                events=[BaseEvent(id="e2", type=EventType.AGENT_SPEAK, source="dog_01",
                                  location_id="kitchen", payload={"message": "Wuff!"},
                                  causality_depth=1, caused_by="e1")],
            ),
        ],
    )


class TestBuildReport:
    def test_is_self_contained_html(self, trace):
        html = build_report_html(trace)
        assert "<html" in html.lower()
        # Plotly-Bibliothek ist eingebettet (offline lauffähig, kein CDN)
        assert "Plotly" in html

    def test_contains_all_four_sections(self, trace):
        html = build_report_html(trace)
        assert "Lebenskurven" in html
        assert "Reaktionsketten" in html
        assert "Tick-Protokoll" in html
        assert "Raumwechsel" in html

    def test_contains_agent_and_messages(self, trace):
        html = build_report_html(trace)
        assert "Whiskers" in html
        assert "Wo ist meine Milch?" in html

    def test_empty_trace_does_not_crash(self):
        empty = Trace(meta=TraceMeta(model="m", agent_ids=[]), ticks=[])
        html = build_report_html(empty)
        assert "<html" in html.lower()


class TestWriteReport:
    def test_writes_file_from_trace(self, tmp_path, trace):
        trace_path = tmp_path / "run.jsonl"
        writer = TraceWriter(trace_path, model="m", agent_ids=["cat_01", "dog_01"])
        for t in trace.ticks:
            writer.record_tick(t)
        writer.close()

        out = tmp_path / "report.html"
        write_report(trace_path, out)
        assert out.exists()
        assert "<html" in out.read_text(encoding="utf-8").lower()
