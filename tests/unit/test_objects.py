"""Tests für Objekt-Effekte (use wirkt auf die Stats)."""

from nanosim.models import AgentStats
from nanosim.world.objects import OBJECT_EFFECTS, apply_object_effect


class TestApplyObjectEffect:
    def test_futternapf_reduces_hunger(self):
        stats = AgentStats(hunger=0.8)
        result = apply_object_effect(stats, "futternapf")
        assert result.hunger < 0.8

    def test_effect_is_clamped_at_zero(self):
        stats = AgentStats(hunger=0.2)
        result = apply_object_effect(stats, "futternapf")
        assert result.hunger == 0.0

    def test_effect_is_clamped_at_one(self):
        stats = AgentStats(stamina=0.95)
        result = apply_object_effect(stats, "sofa")  # erhöht stamina
        assert result.stamina == 1.0

    def test_unknown_object_leaves_stats_unchanged(self):
        stats = AgentStats(stamina=0.5, mood=0.5, hunger=0.5)
        result = apply_object_effect(stats, "fenster")  # kein Effekt hinterlegt
        assert result.stamina == 0.5
        assert result.mood == 0.5
        assert result.hunger == 0.5

    def test_does_not_mutate_original(self):
        stats = AgentStats(hunger=0.8)
        apply_object_effect(stats, "futternapf")
        assert stats.hunger == 0.8  # Original unverändert

    def test_all_effects_reference_valid_stats(self):
        valid = set(AgentStats.model_fields)
        for obj, effect in OBJECT_EFFECTS.items():
            for stat in effect:
                assert stat in valid, f"{obj}: unbekanntes Stat {stat}"
