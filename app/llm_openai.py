"""Bucla de function calling cu OpenAI (SDK openai, Responses API).

Bucla e manuală, ca fiecare apel de tool să fie înregistrat (trasabilitatea cifrelor din raport). Ieșirea
modelului (inclusiv elementele de raționament) se adaugă înapoi în input neschimbată, cum cere API-ul.
"""
from __future__ import annotations

import json
from typing import Callable

import openai
from openai import OpenAI

from .config import settings
from .llm import LLMUnavailable


def _tools(tools: dict) -> list[dict]:
    # strict=False: schemele au parametri opționali (modul strict ar cere toate câmpurile obligatorii)
    return [{"type": "function", "name": name, "description": spec.description,
             "parameters": spec.parameters, "strict": False}
            for name, spec in tools.items()]


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise LLMUnavailable(
            "Agentul AI nu e configurat: setați OPENAI_API_KEY în fișierul .env "
            "(https://platform.openai.com/api-keys) și reporniți API-ul.")
    # reîncercări automate ale SDK-ului pentru 408/409/429/5xx și erori de conexiune
    return OpenAI(api_key=settings.openai_api_key, max_retries=4, timeout=120.0)


def _api_error(e: Exception) -> LLMUnavailable:
    if isinstance(e, openai.AuthenticationError):
        return LLMUnavailable("Cheia OPENAI_API_KEY e invalidă. Verificați .env.")
    if isinstance(e, openai.RateLimitError):
        body = str(getattr(e, "body", "") or e).lower()
        if "insufficient_quota" in body or "quota" in body:
            return LLMUnavailable("Contul OpenAI nu mai are credit (insufficient_quota). Verificați facturarea "
                                  "pe platform.openai.com.")
        return LLMUnavailable("Limita de cereri OpenAI a fost depășită. Reîncercați peste un minut.")
    if isinstance(e, openai.NotFoundError):
        return LLMUnavailable("Modelul OpenAI cerut nu e disponibil pentru acest cont. Verificați OPENAI_MODEL.",
                              temporar=True)
    if isinstance(e, openai.BadRequestError):
        return LLMUnavailable(f"Cerere respinsă de OpenAI: {getattr(e, 'message', e)}")
    if isinstance(e, openai.APITimeoutError):
        return LLMUnavailable("OpenAI nu a răspuns la timp. Reîncercați.", temporar=True)
    if isinstance(e, openai.APIConnectionError):
        return LLMUnavailable("Nu s-a putut conecta la OpenAI (rețea).")
    if isinstance(e, openai.APIStatusError):
        code = e.status_code
        return LLMUnavailable(f"Eroare OpenAI ({code}): {getattr(e, 'message', e)}", temporar=code >= 500)
    return LLMUnavailable(f"Eroare OpenAI: {type(e).__name__}: {e}")


def _extra() -> dict:
    return {"reasoning": {"effort": settings.openai_reasoning_effort}} if settings.openai_reasoning_effort else {}


def generate_text(system: str, prompt: str, model: str | None = None) -> tuple[str, dict]:
    """Un singur apel fără tool-uri. Întoarce (text, consum de tokeni)."""
    client = _client()
    model = model or settings.openai_model
    try:
        r = client.responses.create(model=model, instructions=system, input=prompt,
                                    max_output_tokens=settings.agent_max_output_tokens, **_extra())
    except openai.OpenAIError as e:
        raise _api_error(e) from e
    text = (r.output_text or "").strip()
    trunchiat = r.status == "incomplete"
    if not text:
        raise LLMUnavailable("OpenAI nu a produs raportul (limita de tokeni de ieșire atinsă înainte de text). "
                             "Măriți AGENT_MAX_OUTPUT_TOKENS sau reduceți OPENAI_REASONING_EFFORT."
                             if trunchiat else f"OpenAI a întors un răspuns gol (status: {r.status}).")
    if trunchiat:
        text += "\n\n*(Raport trunchiat: s-a atins limita de tokeni de ieșire.)*"
    u = r.usage
    usage = {"intrare": u.input_tokens, "iesire": u.output_tokens,
             "rationament": getattr(u.output_tokens_details, "reasoning_tokens", None), "total": u.total_tokens} if u else None
    return text, usage


def run_tool_loop(system: str, prompt: str, tools: dict, execute: Callable[[str, dict], dict],
                  max_steps: int | None = None, model: str | None = None) -> str:
    """Rulează conversația până când modelul răspunde cu text. execute(nume, argumente) -> dict (trimis modelului)."""
    client = _client()
    model = model or settings.openai_model
    max_steps = max_steps or settings.agent_max_pasi
    tool_defs = _tools(tools)
    extra = _extra()
    items: list = [{"role": "user", "content": prompt}]
    try:
        for step in range(max_steps + 1):
            last = step == max_steps
            response = client.responses.create(
                model=model, instructions=system, input=items, tools=tool_defs,
                tool_choice="none" if last else "auto",  # ultima rundă: doar raportul
                **extra,
            )
            items += response.output  # neschimbat: păstrează elementele de raționament
            calls = [it for it in response.output if it.type == "function_call"]
            if not calls:
                text = (response.output_text or "").strip()
                if not text:
                    status = getattr(response, "status", None)
                    raise LLMUnavailable(f"OpenAI a întors un răspuns gol (status: {status}).")
                return text
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    result = {"eroare": "Argumentele apelului nu sunt JSON valid."}
                else:
                    result = execute(call.name, args if isinstance(args, dict) else {})
                items.append({"type": "function_call_output", "call_id": call.call_id,
                              "output": json.dumps(result, ensure_ascii=False, default=str)})
            if step == max_steps - 1:
                items.append({"role": "user", "content":
                              "Ai atins numărul maxim de pași. Scrie acum raportul final doar cu datele obținute."})
    except LLMUnavailable:
        raise
    except openai.OpenAIError as e:
        raise _api_error(e) from e
    except Exception as e:  # răspuns neașteptat
        raise LLMUnavailable(f"Eroare la comunicarea cu OpenAI: {type(e).__name__}: {e}") from e
    raise LLMUnavailable("Agentul nu a produs un raport.")
