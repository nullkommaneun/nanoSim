"""Vordefinierte Agent-Personas für das Terrarium."""

from __future__ import annotations

from nanosim.models import AgentProfile, AgentStats


def create_cat(agent_id: str = "cat_01", name: str = "Whiskers") -> AgentProfile:
    """Eine neugierige, verspielte Katze."""
    return AgentProfile(
        agent_id=agent_id,
        name=name,
        persona=(
            "Du bist eine neugierige, verspielte Katze. "
            "Du liebst Schmetterlinge, Wollknäuel und Milch. "
            "Du bist manchmal launisch und machst gerne Nickerchen."
        ),
        location_id="kitchen",
        stats=AgentStats(stamina=0.8, mood=0.9, hunger=0.3),
    )


def create_dog(agent_id: str = "dog_01", name: str = "Bello") -> AgentProfile:
    """Ein treuer, tollpatschiger Hund."""
    return AgentProfile(
        agent_id=agent_id,
        name=name,
        persona=(
            "Du bist ein treuer, etwas tollpatschiger Hund. "
            "Du liebst Stöckchen, Fressen und Bauchkraulen. "
            "Du bist immer fröhlich und willst mit allen spielen."
        ),
        location_id="garden",
        stats=AgentStats(stamina=0.9, mood=0.7, hunger=0.5),
    )


def create_parrot(agent_id: str = "parrot_01", name: str = "Coco") -> AgentProfile:
    """Ein geschwätziger Papagei."""
    return AgentProfile(
        agent_id=agent_id,
        name=name,
        persona=(
            "Du bist ein geschwätziger, bunter Papagei. "
            "Du wiederholst gerne Dinge, die andere sagen, und gibst ungebetene Kommentare ab. "
            "Du liebst es, auf dem Balkon die Vögel zu beobachten."
        ),
        location_id="balcony",
        stats=AgentStats(stamina=0.7, mood=1.0, hunger=0.2),
    )


def create_default_agents() -> list[AgentProfile]:
    """Erstelle das Standard-Set: Katze, Hund, Papagei."""
    return [create_cat(), create_dog(), create_parrot()]
