"""Bucla de function calling cu Google Gemini (SDK google-genai, API generate_content).

Bucla e manuală (fără automatic function calling), ca fiecare apel de tool să fie înregistrat: rapoartele trebuie
să arate de unde vine fiecare cifră. Conținutul modelului se adaugă înapoi în istoric NESCHIMBAT, pentru că
modelele Gemini 3 cer ca thought signatures să fie retrimise.
"""
from __future__ import annotations

import json
from typing import Callable

from google import genai
from google.genai import errors, types

from .config import settings
from .llm import LLMUnavailable


# reîncercări automate ale SDK-ului pentru erori trecătoare (limită/minut, supraîncărcare)
RETRY = types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=20.0, exp_base=2.0,
                               http_status_codes=[429, 500, 503, 504])


def _declarations(tools: dict) -> list[types.Tool]:
    decls = [
        types.FunctionDeclaration(name=name, description=spec.description, parameters_json_schema=spec.parameters)
        for name, spec in tools.items()
    ]
    return [types.Tool(function_declarations=decls)]


def _client() -> genai.Client:
    if not settings.gemini_api_key:
        raise LLMUnavailable(
            "Agentul AI nu e configurat: setați GEMINI_API_KEY în fișierul .env (cheie gratuită de la "
            "https://aistudio.google.com) și reporniți API-ul.")
    return genai.Client(api_key=settings.gemini_api_key)


def _api_error(e: errors.APIError) -> LLMUnavailable:
    code = getattr(e, "code", None)
    if code == 429:
        return LLMUnavailable("Limita gratuită Gemini a fost depășită (prea multe cereri). Reîncercați peste un minut.")
    if code in (500, 503, 504):
        return LLMUnavailable(f"Modelul Gemini e momentan supraîncărcat ({code}). Reîncercați peste câteva momente.",
                              temporar=True)
    if code in (400, 401, 403) and "key" in str(e).lower():
        return LLMUnavailable("Cheia GEMINI_API_KEY e invalidă sau nu are acces la model. Verificați .env.")
    if code == 404:
        return LLMUnavailable("Modelul Gemini cerut nu e disponibil. Verificați GEMINI_MODEL în .env.", temporar=True)
    return LLMUnavailable(f"Eroare de la Gemini ({code}): {getattr(e, 'message', None) or e}")


def generate_text(system: str, prompt: str, model: str | None = None) -> tuple[str, dict]:
    """Un singur apel fără tool-uri. Întoarce (text, consum de tokeni)."""
    client = _client()
    cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=settings.agent_max_output_tokens,
                                      http_options=types.HttpOptions(timeout=120_000, retry_options=RETRY))
    try:
        r = client.models.generate_content(model=model or settings.gemini_model, contents=prompt, config=cfg)
    except errors.APIError as e:
        raise _api_error(e) from e
    except Exception as e:
        raise LLMUnavailable(f"Nu s-a putut comunica cu Gemini: {type(e).__name__}: {e}") from e
    text = (r.text or "").strip() if r.candidates else ""
    trunchiat = bool(r.candidates) and r.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS
    if not text:
        raise LLMUnavailable("Gemini nu a produs raportul" + (" (limita de tokeni de ieșire)." if trunchiat else "."))
    if trunchiat:
        text += "\n\n*(Raport trunchiat: s-a atins limita de tokeni de ieșire.)*"
    m = r.usage_metadata
    usage = {"intrare": m.prompt_token_count, "iesire": m.candidates_token_count,
             "rationament": m.thoughts_token_count, "total": m.total_token_count} if m else None
    return text, usage


def run_tool_loop(system: str, prompt: str, tools: dict, execute: Callable[[str, dict], dict],
                  max_steps: int | None = None, model: str | None = None) -> str:
    """Rulează conversația până când modelul răspunde cu text. execute(nume, argumente) -> dict (trimis modelului).

    Întoarce textul final. Ridică LLMUnavailable pentru orice problemă cu modelul de limbaj.
    """
    client = _client()
    model = model or settings.gemini_model
    max_steps = max_steps or settings.agent_max_pasi
    base_cfg = dict(system_instruction=system, tools=_declarations(tools),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    http_options=types.HttpOptions(timeout=120_000, retry_options=RETRY))
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    try:
        for step in range(max_steps + 1):
            last = step == max_steps
            cfg = types.GenerateContentConfig(
                **base_cfg,
                # ultima rundă: fără tool-uri, modelul trebuie să scrie raportul cu ce are
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.NONE)) if last else None,
            )
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            if not response.candidates:
                feedback = getattr(response, "prompt_feedback", None)
                raise LLMUnavailable(f"Gemini nu a întors niciun răspuns ({feedback}).")
            calls = response.function_calls or []
            if not calls:
                text = (response.text or "").strip()
                if not text:
                    raise LLMUnavailable("Gemini a întors un răspuns gol.")
                return text
            contents.append(response.candidates[0].content)  # neschimbat: păstrează thought signatures
            parts = []
            for call in calls:
                result = execute(call.name, dict(call.args or {}))
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=call.id, name=call.name, response={"result": json.loads(json.dumps(result, default=str))})))
            contents.append(types.Content(role="user", parts=parts))
            if step == max_steps - 1:
                contents.append(types.Content(role="user", parts=[types.Part(
                    text="Ai atins numărul maxim de pași. Scrie acum raportul final doar cu datele obținute.")]))
    except LLMUnavailable:
        raise
    except errors.APIError as e:
        raise _api_error(e) from e
    except Exception as e:  # rețea, timeout, răspuns neașteptat
        raise LLMUnavailable(f"Nu s-a putut comunica cu Gemini: {type(e).__name__}: {e}") from e
    raise LLMUnavailable("Agentul nu a produs un raport.")
