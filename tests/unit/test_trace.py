"""Tests für den TraceWriter / das Trace-Datenmodell (Aufzeichnung)."""

import json

from nanosim.models import ActionType, AgentStats, BaseEvent, EventType
from nanosim.trace import (
    AgentSnapshot,
    Decision,
    TickRecord,
    TraceWriter,
    load_trace,
)


def _sample_tick(tick: int) -> TickRecord:
    return TickRecord(
        tick=tick,
        agents=[
            AgentSnapshot(
                agent_id="cat_01", name="Whiskers", location_id="kitchen",
                stats=AgentStats(stamina=0.8, mood=0.7, hunger=0.3),
            ),
        ],
        decisions=[
            Decision(agent_id="cat_01", action=ActionType.SPEAK, message="Miau!"),
        ],
        events=[
            BaseEvent(
                id="evt1", type=EventType.AGENT_SPEAK, source="cat_01",
                location_id="kitchen", payload={"message": "Miau!"},
                causality_depth=2, caused_by="evt0",
            ),
        ],
    )


class TestTraceWriter:
    def test_writes_meta_and_ticks(self, tmp_path):
        path = tmp_path / "run.jsonl"
        writer = TraceWriter(path, model="llama3.1:8b", agent_ids=["cat_01"])
        writer.record_tick(_sample_tick(0))
        writer.record_tick(_sample_tick(1))
        writer.close()

        trace = load_trace(path)
        assert trace.meta.model == "llama3.1:8b"
        assert trace.meta.agent_ids == ["cat_01"]
        assert [t.tick for t in trace.ticks] == [0, 1]

    def test_jsonl_one_valid_json_per_line(self, tmp_path):
        path = tmp_path / "run.jsonl"
        writer = TraceWriter(path, model="m", agent_ids=["cat_01"])
        writer.record_tick(_sample_tick(0))
        writer.close()

        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2  # 1 Meta-Zeile + 1 Tick-Zeile
        for ln in lines:
            json.loads(ln)  # jede Zeile ist valides JSON

    def test_roundtrip_preserves_decisions(self, tmp_path):
        path = tmp_path / "run.jsonl"
        writer = TraceWriter(path, model="m", agent_ids=["cat_01"])
        writer.record_tick(_sample_tick(0))
        writer.close()

        trace = load_trace(path)
        dec = trace.ticks[0].decisions[0]
        assert dec.agent_id == "cat_01"
        assert dec.action == ActionType.SPEAK
        assert dec.message == "Miau!"

    def test_roundtrip_preserves_event_causality(self, tmp_path):
        path = tmp_path / "run.jsonl"
        writer = TraceWriter(path, model="m", agent_ids=["cat_01"])
        writer.record_tick(_sample_tick(0))
        writer.close()

        trace = load_trace(path)
        ev = trace.ticks[0].events[0]
        assert ev.caused_by == "evt0"
        assert ev.causality_depth == 2

    def test_roundtrip_preserves_stats(self, tmp_path):
        path = tmp_path / "run.jsonl"
        writer = TraceWriter(path, model="m", agent_ids=["cat_01"])
        writer.record_tick(_sample_tick(0))
        writer.close()

        trace = load_trace(path)
        snap = trace.ticks[0].agents[0]
        assert snap.name == "Whiskers"
        assert snap.stats.hunger == 0.3
