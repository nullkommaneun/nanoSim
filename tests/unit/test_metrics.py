"""Tests für die Lebendigkeits-Kennzahlen (Analyse eines Traces)."""

from nanosim.metrics import aggregate, compute_metrics
from nanosim.models import ActionType, AgentStats, BaseEvent, EventType
from nanosim.trace import (
    AgentSnapshot,
    Decision,
    TickRecord,
    Trace,
    TraceMeta,
)


def _snap(aid, room):
    return AgentSnapshot(agent_id=aid, name=aid, location_id=room, stats=AgentStats())


def _trace(ticks):
    return Trace(meta=TraceMeta(model="m", agent_ids=["a", "b"]), ticks=ticks)


class TestComputeMetrics:
    def test_reaction_chains_counts_caused_by_speaks(self):
        t = _trace(
            [
                TickRecord(
                    tick=0,
                    agents=[_snap("a", "k"), _snap("b", "k")],
                    decisions=[
                        Decision(agent_id="a", action=ActionType.SPEAK, message="hi")
                    ],
                    events=[
                        BaseEvent(
                            id="e1",
                            type=EventType.AGENT_SPEAK,
                            source="a",
                            payload={"message": "hi"},
                        ),
                        BaseEvent(
                            id="e2",
                            type=EventType.AGENT_SPEAK,
                            source="b",
                            payload={"message": "yo"},
                            caused_by="e1",
                        ),
                    ],
                )
            ]
        )
        assert compute_metrics(t)["reaction_chains"] == 1

    def test_togetherness_and_avg_room(self):
        t = _trace(
            [
                TickRecord(
                    tick=0,
                    agents=[_snap("a", "k"), _snap("b", "k")],
                    decisions=[],
                    events=[],
                ),
                TickRecord(
                    tick=1,
                    agents=[_snap("a", "k"), _snap("b", "g")],
                    decisions=[],
                    events=[],
                ),
            ]
        )
        m = compute_metrics(t)
        assert m["togetherness"] == 0.5  # 1 von 2 Ticks hatte >=2 zusammen
        assert m["avg_max_room"] == 1.5  # Tick0 max=2, Tick1 max=1

    def test_action_counts(self):
        t = _trace(
            [
                TickRecord(
                    tick=0,
                    agents=[_snap("a", "k")],
                    decisions=[
                        Decision(agent_id="a", action=ActionType.MOVE, target="north"),
                        Decision(agent_id="a", action=ActionType.USE, target="napf"),
                    ],
                    events=[],
                )
            ]
        )
        m = compute_metrics(t)
        assert m["moves"] == 1
        assert m["uses"] == 1

    def test_unique_message_ratio(self):
        t = _trace(
            [
                TickRecord(
                    tick=0,
                    agents=[_snap("a", "k")],
                    decisions=[
                        Decision(
                            agent_id="a", action=ActionType.SPEAK, message="hallo"
                        ),
                        Decision(
                            agent_id="b", action=ActionType.SPEAK, message="hallo"
                        ),
                    ],
                    events=[],
                )
            ]
        )
        m = compute_metrics(t)
        assert m["speaks"] == 2
        assert m["unique_message_ratio"] == 0.5  # 1 eindeutige / 2

    def test_empty_trace_does_not_crash(self):
        m = compute_metrics(_trace([]))
        assert m["reaction_chains"] == 0
        assert m["togetherness"] == 0.0


class TestAggregate:
    def test_mean_and_std(self):
        agg = aggregate([{"reaction_chains": 2}, {"reaction_chains": 4}])
        assert agg["reaction_chains"]["mean"] == 3.0
        assert agg["reaction_chains"]["std"] > 0

    def test_single_value_zero_std(self):
        agg = aggregate([{"reaction_chains": 5}])
        assert agg["reaction_chains"]["mean"] == 5.0
        assert agg["reaction_chains"]["std"] == 0.0
