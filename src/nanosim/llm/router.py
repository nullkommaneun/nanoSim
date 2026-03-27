"""VRAM-Türsteher: Serialisierter Zugang zu Ollama mit JSON-Reparatur."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Suffix der an jeden Prompt angehängt wird
_JSON_INSTRUCTION = (
    "\n\nRespond ONLY with valid JSON matching this schema:\n"
    "{schema}\n\n"
    "Do not include any text before or after the JSON."
)

_RETRY_INSTRUCTION = (
    "Your last response was not valid JSON. Error: {error}\n\n"
    "Respond ONLY with valid JSON matching this schema:\n"
    "{schema}\n\n"
    "Do not include any text before or after the JSON."
)


class LlamaRouter:
    """Serialisierter Ollama-Zugang mit Semaphore(1) und Auto-Retry.

    Garantiert, dass nur ein LLM-Call gleichzeitig an Ollama geht.
    Bei JSON/Validation-Fehlern wird ein automatischer Retry mit
    Fehlerfeedback an das Modell gestartet.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._client = ollama.AsyncClient(host=base_url)
        self._semaphore = asyncio.Semaphore(1)
        self.model = model

    async def think(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
    ) -> T | None:
        """Sende einen Prompt an Ollama und validiere die Antwort.

        Args:
            prompt: Der User-Prompt.
            response_model: Pydantic-Modell gegen das validiert wird.
            system: Optionaler System-Prompt.

        Returns:
            Eine validierte Instanz von response_model, oder None bei Fehler.
        """
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        full_prompt = prompt + _JSON_INSTRUCTION.format(schema=schema_str)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": full_prompt})

        # --- Versuch 1 ---
        raw = await self._call_ollama(messages)
        result = self._parse_and_validate(raw, response_model)
        if result is not None:
            return result

        # --- Versuch 2: Retry mit Fehlerfeedback ---
        logger.warning("Erster Versuch fehlgeschlagen, starte Retry für %s", self.model)
        error_msg = self._get_parse_error(raw, response_model)
        retry_prompt = _RETRY_INSTRUCTION.format(error=error_msg, schema=schema_str)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": retry_prompt})

        raw = await self._call_ollama(messages)
        result = self._parse_and_validate(raw, response_model)
        if result is not None:
            return result

        logger.error("Retry fehlgeschlagen. Agent wird idle.")
        return None

    async def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        """Einen Chat-Call an Ollama senden, geschützt durch den Semaphore."""
        async with self._semaphore:
            response = await self._client.chat(
                model=self.model,
                messages=messages,
            )
        return response["message"]["content"]

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Versuche JSON aus der Antwort zu extrahieren.

        Modelle wrappen JSON manchmal in Markdown-Codeblöcke oder
        schreiben Text davor/danach. Wir suchen das erste { und letzte }.
        """
        # Markdown-Codeblock entfernen
        if "```json" in raw:
            raw = raw.split("```json", 1)[1]
            raw = raw.split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1]
            raw = raw.split("```", 1)[0]

        # Erstes { bis letztes } extrahieren
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        return raw.strip()

    @staticmethod
    def _parse_and_validate(raw: str, model: type[T]) -> T | None:
        """JSON parsen und gegen Pydantic-Modell validieren."""
        cleaned = LlamaRouter._extract_json(raw)
        try:
            data = json.loads(cleaned)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            return None

    @staticmethod
    def _get_parse_error(raw: str, model: type[T]) -> str:
        """Fehlertext für den Retry-Prompt generieren."""
        cleaned = LlamaRouter._extract_json(raw)
        try:
            data = json.loads(cleaned)
            model.model_validate(data)
            return "Unknown error"
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"
        except ValidationError as e:
            return f"Schema validation failed: {e}"
