"""Tests für das gehärtete Experiment-Labor (nanosim.lab)."""

import pytest

from nanosim.lab.config import EXPERIMENTS, active_configs, build_cmd
from nanosim.lab.record import RunRecord, append_record, load_records
from nanosim.lab.report import render_report
from nanosim.lab.stats import bootstrap_ci, rank_configs


def _rec(config, rc, status="ok", **kw):
    return RunRecord(
        round=1, config=config, ts=0.0, status=status,
        metrics={"reaction_chains": rc, "togetherness": 1.0,
                 "avg_max_room": 2.0, "unique_message_ratio": 0.8, "moves": 1},
        **kw,
    )


class TestConfig:
    def test_active_all_is_twelve(self):
        assert len(active_configs("all")) == 12

    def test_active_named(self):
        assert len(active_configs("models")) == 5

    def test_active_unknown_raises_hard(self):
        with pytest.raises(ValueError):
            active_configs("bogus")

    def test_build_cmd_flags(self):
        cfg = {"name": "x", "model": "m", "dialogue": "d", "world": "hub5",
               "prompt_variant": "hard", "memory_window": 6}
        cmd = build_cmd(cfg, "/tmp/t.jsonl", seed=42)
        joined = " ".join(cmd)
        for frag in ["--model m", "--dialogue-model d", "--world hub5",
                     "--prompt-variant hard", "--memory-window 6", "--seed 42"]:
            assert frag in joined

    def test_experiments_registry(self):
        assert {"models", "prompt", "world", "memory"} <= set(EXPERIMENTS)


class TestRecords:
    def test_roundtrip_and_skip_malformed(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        append_record(path, _rec("a", 5))
        # kaputte Zeile manuell anhängen
        with path.open("a", encoding="utf-8") as f:
            f.write("{ das ist kein JSON \n")
        append_record(path, _rec("a", 7))

        records, skipped = load_records(path)
        assert len(records) == 2
        assert skipped == 1

    def test_missing_file(self, tmp_path):
        records, skipped = load_records(tmp_path / "nix.jsonl")
        assert records == [] and skipped == 0


class TestStats:
    def test_bootstrap_ci_deterministic(self):
        vals = [10, 12, 11, 13, 9, 10, 12]
        a = bootstrap_ci(vals)
        b = bootstrap_ci(vals)
        assert a == b  # fester Seed → reproduzierbar
        assert a[0] <= sum(vals) / len(vals) <= a[1]

    def test_degenerate_excluded_from_ranking(self):
        records = (
            [_rec("good", 20) for _ in range(5)]
            + [_rec("dead", 0) for _ in range(5)]
        )
        ranked, degenerate = rank_configs(records)
        names = {e.name for e in ranked}
        assert "good" in names
        assert "dead" not in names
        assert any(e.name == "dead" for e in degenerate)

    def test_success_rate_counts_failures(self):
        records = [_rec("c", 5), _rec("c", 5), _rec("c", 0, status="timeout")]
        ranked, _ = rank_configs(records)
        entry = next(e for e in ranked if e.name == "c")
        assert entry.n_total == 3
        assert entry.n_ok == 2  # nur ok-Läufe zählen für die Kennzahl

    def test_ties_marked_when_ci_overlaps(self):
        # zwei nahezu identische Verteilungen → überlappende KIs
        records = (
            [_rec("a", v) for v in [20, 21, 19, 20, 22, 18]]
            + [_rec("b", v) for v in [20, 19, 21, 20, 18, 22]]
        )
        ranked, _ = rank_configs(records)
        assert all(e.tied_with_top for e in ranked)


class TestReport:
    def test_render_contains_health_and_counts(self):
        records = [_rec("good", 20) for _ in range(4)] + [_rec("dead", 0) for _ in range(4)]
        md = render_report(records, meta={"experiment": "world", "health": "GRÜN"})
        assert "world" in md
        assert "GRÜN" in md
        assert "degeneriert" in md  # tote Config wird als degeneriert ausgewiesen
        assert "`good`" in md

    def test_render_empty_does_not_crash(self):
        md = render_report([])
        assert "Report" in md
