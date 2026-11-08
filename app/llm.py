"""Stratul comun pentru modelul de limbaj al agentului: eroarea comună și alegerea furnizorului.

LLM_PROVIDER din .env alege implementarea: "openai" (implicit, app/llm_openai.py) sau "gemini"
(app/llm_gemini.py). Ambele expun run_tool_loop(system, prompt, tools, execute, model=...) -> text.
"""
from __future__ import annotations

from typing import Callable

from .config import settings


class LLMUnavailable(RuntimeError):
    """Modelul de limbaj nu poate fi folosit (cheie lipsă/invalidă, cotă depășită, eroare de rețea).

    temporar=True: problemă trecătoare de partea furnizorului (supraîncărcare, model indisponibil),
    merită încercat modelul de rezervă."""

    def __init__(self, message: str, temporar: bool = False):
        super().__init__(message)
        self.temporar = temporar


def provider() -> str:
    return (settings.llm_provider or "openai").strip().lower()


def models() -> list[str]:
    """Modelul principal și (opțional) cel de rezervă ale furnizorului activ."""
    if provider() == "gemini":
        main, backup = settings.gemini_model, settings.gemini_model_rezerva
    else:
        main, backup = settings.openai_model, settings.openai_model_rezerva
    return [main] + ([backup] if backup and backup != main else [])


def is_configured() -> bool:
    return bool(settings.gemini_api_key if provider() == "gemini" else settings.openai_api_key)


def provider_label() -> str:
    return "Google Gemini (google-genai)" if provider() == "gemini" else "OpenAI (Responses API)"


def run_tool_loop(system: str, prompt: str, tools: dict, execute: Callable[[str, dict], dict],
                  model: str | None = None, max_steps: int | None = None) -> str:
    p = provider()
    if p == "gemini":
        from .llm_gemini import run_tool_loop as impl
    elif p == "openai":
        from .llm_openai import run_tool_loop as impl
    else:
        raise LLMUnavailable(f"LLM_PROVIDER necunoscut: „{p}”. Valori acceptate: openai, gemini.")
    return impl(system, prompt, tools, execute, max_steps=max_steps, model=model)
