"""Zwei-Stufen-Router: kleines Modell entscheidet, großes Modell spricht.

Etabliertes Muster (Model Cascading/Routing): Die billige, strukturierte
Entscheidung (welche Aktion?) trifft ein kleines, schnelles Modell. Nur wenn
gesprochen wird, formuliert ein großes Modell die eigentliche Rede — teuer,
aber selten. So wird die knappe Rechenzeit gezielt dort eingesetzt, wo sie die
Qualität wirklich hebt. An die Hardware anpassbar über die Modellwahl.

Der TwoTierRouter ist Drop-in-kompatibel zum LlamaRouter: gleiche
`think(prompt, response_model, system)`-Signatur, gibt eine AgentAction zurück.
"""

from __future__ import annotations

from nanosim.llm.router import DEFAULT_MODEL, LlamaRouter
from nanosim.models import ActionType, AgentAction, DecisionAction, SpeechLine

# An den Kontext-Prompt angehängt, wenn das große Modell die Rede formulieren soll.
_SPEAK_INSTRUCTION = (
    "\n\nDu hast dich entschieden zu SPRECHEN. Formuliere kurz, lebendig und "
    "ganz in deiner Rolle, was du jetzt laut sagst."
)


class TwoTierRouter:
    """Router mit zwei Modellen: eines entscheidet, eines spricht."""

    def __init__(
        self,
        decision_model: str = DEFAULT_MODEL,
        dialogue_model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        seed: int | None = None,
    ) -> None:
        self._decider = LlamaRouter(model=decision_model, base_url=base_url, seed=seed)
        self._talker = LlamaRouter(model=dialogue_model, base_url=base_url, seed=seed)
        # Anzeigename für Logs/Trace
        self.model = f"{decision_model} (Entscheidung) + {dialogue_model} (Dialog)"

    async def think(
        self,
        prompt: str,
        response_model: type | None = None,
        system: str | None = None,
    ) -> AgentAction | None:
        """Erst entscheiden (klein), bei 'speak' die Rede formulieren (groß).

        `response_model` wird ignoriert — der Router liefert immer eine
        AgentAction (so bleibt er Drop-in-kompatibel zum LlamaRouter).
        """
        decision = await self._decider.think(
            prompt,
            DecisionAction,
            system=system,
        )
        if decision is None:
            return None

        if decision.action != ActionType.SPEAK:
            return AgentAction(action=decision.action, target=decision.target)

        # Sprechen → das große Modell formuliert die Rede.
        line = await self._talker.think(
            prompt + _SPEAK_INSTRUCTION,
            SpeechLine,
            system=system,
        )
        # Versagt das Dialog-Modell (keine/leere Rede) → idle statt leeres speak,
        # damit die aufgezeichnete Entscheidung zur ausgeführten Aktion passt.
        if line is None or not line.message.strip():
            return AgentAction(action=ActionType.IDLE)
        return AgentAction(
            action=ActionType.SPEAK,
            target=decision.target,
            message=line.message,
        )
