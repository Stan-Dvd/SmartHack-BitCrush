"""Agentul AI: transformă o acțiune din meniu într-o sarcină pentru LLM, execută tool-urile cerute de model și
asamblează harta DETERMINIST din rezultatele tool-urilor (nu din textul modelului).

Principiul: LLM-ul conduce (alege tool-urile, explică), tool-urile calculează. Toate cifrele vin din tool-uri.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .agent_tools import TOOLS, ToolError, for_llm, run_tool
from .config import settings
from .db import get_connection
from .llm_gemini import LLMUnavailable, run_tool_loop

SYSTEM = """Ești un analist de infrastructură de încărcare pentru vehicule electrice în România.
Ajuți la alegerea locațiilor pentru stații noi folosind EXCLUSIV tool-urile disponibile.

Reguli obligatorii:
- Toate cifrele din răspuns (ore de încărcare, ocupare, kWh, canibalizare, bani, populație) trebuie să provină
  din rezultatele tool-urilor. Nu estima, nu rotunji agresiv și nu inventa valori. Dacă o informație lipsește,
  spune că lipsește.
- Nu scrie SQL și nu face calcule proprii de profit: pentru bani folosește tool-ul economics.
- Nu recomanda numărul de puncte (n_piles) pe baza modelului: efectul lui învățat nu e cauzal. Folosește
  configurația primită în sarcină.
- Modelul nu prezice saturarea; nu face afirmații despre cozi sau stații pline.
- Dacă un tool întoarce o eroare sau o zonă ambiguă, explică problema în raport în loc să ghicești.
- Spune clar când o locație e la „nivelul de bază” al modelului (peste_nivelul_de_baza = false): atunci ordinea
  vine din criteriul secundar (populație, POI), nu dintr-o cerere prezisă mai mare.
- Menționează limitele relevante: model antrenat pe Shenzhen (nivelurile absolute sunt orientative, utile pentru
  comparații), populație WorldPop suspectă (pop_suspect), ipoteze economice ilustrative.

Numerele se scriu în format românesc: virgulă zecimală (938,9 kWh/zi; 18,78 ore), fără punct ca separator
de mii (scrie 202509 kWh/an sau 202 509 kWh/an). kWh/zi prezis e un plafon (puncte ocupate x putere nominală),
nu energie livrată efectiv: energia livrată și banii vin din tool-ul economics.

Formatul răspunsului: markdown în limba română, concis (cel mult ~400 de cuvinte), cu secțiunile cerute în
sarcină. Folosește tabele pentru liste de locații. Nu include JSON brut."""


ACTIUNI = ("sugereaza_statii", "evalueaza_punct", "compara_judete")


def _econ_args(p: dict) -> str:
    keys = ("pret_kwh", "cost_energie_kwh", "capex_per_punct", "opex_procent_capex", "factor_putere_medie",
            "factor_ajustare_cerere")
    vals = {k: p[k] for k in keys if p.get(k) is not None}
    return json.dumps(vals, ensure_ascii=False) if vals else "valorile implicite ale tool-ului"


def build_prompt(actiune: str, p: dict) -> str:
    n = int(p.get("n_piles") or settings.predictie_n_piles_default)
    pw = float(p.get("power_kw") or settings.predictie_power_kw_default)
    if actiune == "sugereaza_statii":
        zona = p.get("zona")
        if not zona:
            raise ValueError("Parametrul „zona” e obligatoriu.")
        k = int(p.get("k") or 3)
        judet = f", judet=„{p['judet']}”" if p.get("judet") else ""
        return f"""Sarcină: propune {k} locații noi pentru stații de încărcare în zona „{zona}”{judet}.
Configurația stației noi: n_piles={n}, power_kw={pw}. Distanța minimă între stațiile propuse: {p.get('min_dist_m', 1000)} m.
Distanța minimă față de stațiile existente: {p.get('min_dist_statie_m', 0)} m.

Pași:
1. resolve_area pentru zonă (dacă e ambiguă sau nu există, oprește-te și explică).
2. get_existing_stations pentru context.
3. plan_network cu k={k}, n_piles={n}, power_kw={pw}, min_dist_m={p.get('min_dist_m', 1000)}, min_dist_statie_m={p.get('min_dist_statie_m', 0)}.
4. explain_cell pentru fiecare locație aleasă (h3_index, aceeași configurație).
5. economics pentru fiecare locație, cu kwh_zi = kwh_zi_estimat și ipotezele: {_econ_args(p)}.

Secțiuni în raport: ## Rezumat; ## Locații recomandate (tabel: #, localizare/județ, ore încărcare/zi,
câștig net ore/zi, kWh/zi, peste nivelul de bază?); ## De ce aceste locații (top contribuții pe locație);
## Impact asupra stațiilor existente; ## Economie (ipoteze ilustrative); ## Limitări."""
    if actiune == "evalueaza_punct":
        lat, lon = p.get("lat"), p.get("lon")
        if lat is None or lon is None:
            raise ValueError("Parametrii lat și lon sunt obligatorii.")
        return f"""Sarcină: evaluează o stație nouă în punctul lat={lat}, lon={lon}, cu n_piles={n}, power_kw={pw}.

Pași:
1. simulate_station în acest punct (aceeași configurație).
2. explain_cell cu lat={lat}, lon={lon} (aceeași configurație).
3. economics cu kwh_zi = sumar.kwh_zi din simulate_station și ipotezele: {_econ_args(p)}.

Secțiuni în raport: ## Verdict (merită / nu merită, comparativ cu percentilele); ## Cererea prezisă;
## De ce (top contribuții); ## Canibalizare; ## Economie (ipoteze ilustrative); ## Limitări."""
    if actiune == "compara_judete":
        return f"""Sarcină: unde în România merită investit în stații noi? Configurație: n_piles={n}, power_kw={pw}.

Pași:
1. rank_areas cu top=10, n_piles={n}, power_kw={pw}.
2. score_candidates pentru primul județ din clasament (top_k=3, aceeași configurație).

Secțiuni în raport: ## Rezumat; ## Clasament (tabel cu primele 10 județe: celule de oportunitate, ore max/zi,
stații existente, stații la 100k locuitori); ## Cel mai bun județ: locații de pornire; ## Cum citim clasamentul
(definiția celulei de oportunitate); ## Limitări."""
    raise ValueError(f"Acțiune necunoscută: {actiune}. Disponibile: {', '.join(ACTIUNI)}")


def _summary(result: Any, limit=280) -> str:
    s = json.dumps(for_llm(result), ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def build_map(calls: list[dict]) -> dict:
    """Straturile hărții, din rezultatele tool-urilor (ultimul rezultat relevant câștigă)."""
    harta: dict[str, Any] = {"contur": None, "bbox": None, "centru": None, "statii": [], "recomandari": [],
                             "afectate": [], "punct": None}
    for c in calls:
        if not c["ok"]:
            continue
        h = (c["_result"] or {}).get("_harta") or {}
        name = c["tool"]
        if name == "resolve_area":
            harta.update(contur=h.get("contur"), bbox=h.get("bbox"), centru=h.get("centru"))
        elif name == "get_existing_stations":
            harta["statii"] = h.get("statii", [])
        elif name in ("plan_network", "rank_areas"):
            harta["recomandari"] = [{"lat": r[0], "lon": r[1], "eticheta": str(r[2]), "id": r[3], "ore": r[4]}
                                    for r in h.get("recomandari", [])]
            if name == "plan_network":
                harta["afectate"] = [{"lat": a[0], "lon": a[1], "cod": a[2], "delta": a[3]} for a in h.get("afectate", [])]
        elif name == "score_candidates" and not harta["recomandari"]:
            harta["recomandari"] = [{"lat": r[0], "lon": r[1], "eticheta": str(i + 1), "id": r[2], "ore": r[3]}
                                    for i, r in enumerate(h.get("candidati", []))]
        elif name == "simulate_station":
            harta["punct"] = h.get("punct")
            harta["afectate"] = [{"lat": a[0], "lon": a[1], "cod": a[2], "delta": a[3]} for a in h.get("afectate", [])]
        elif name == "explain_cell" and harta["punct"] is None and not harta["recomandari"]:
            harta["punct"] = h.get("punct")
    return harta


def run(actiune: str, parametri: dict | None = None) -> dict:
    """Rulează o acțiune din meniu. Ridică LLMUnavailable (fără fallback) sau ValueError (parametri greșiți)."""
    p = dict(parametri or {})
    prompt = build_prompt(actiune, p)
    calls: list[dict] = []
    t0 = time.time()

    def execute(name: str, args: dict) -> dict:
        t = time.time()
        entry = {"tool": name, "argumente": args, "ok": True}
        try:
            with get_connection() as conn:
                result = run_tool(conn, name, args)
            entry.update(_result=result, rezumat=_summary(result))
            payload = for_llm(result)
        except ToolError as e:
            entry.update(ok=False, _result=None, rezumat=f"Eroare: {e}")
            payload = {"eroare": str(e)}
        except Exception as e:  # eroare neașteptată într-un tool: modelul o vede, raportul o menționează
            entry.update(ok=False, _result=None, rezumat=f"Eroare internă: {type(e).__name__}: {e}")
            payload = {"eroare": f"Eroare internă în tool-ul {name}."}
        entry["durata_s"] = round(time.time() - t, 2)
        calls.append(entry)
        return payload

    # modelul principal, apoi (doar la supraîncărcare) cel de rezervă; analiza se reia de la capăt,
    # ca istoricul conversației să conțină un singur model
    modele = [settings.gemini_model] + ([settings.gemini_model_rezerva]
                                         if settings.gemini_model_rezerva and settings.gemini_model_rezerva != settings.gemini_model else [])
    raport, model_folosit = None, None
    for i, model in enumerate(modele):
        calls.clear()
        try:
            raport = run_tool_loop(SYSTEM, prompt, TOOLS, execute, model=model)
            model_folosit = model
            break
        except LLMUnavailable as e:
            if not e.temporar or i == len(modele) - 1:
                raise
    plan = next((c["_result"] for c in reversed(calls) if c["tool"] == "plan_network" and c["ok"]), None)
    salvabile = []
    if plan:
        cfg = plan["configuratie"]
        salvabile = [{"lat": a["lat"], "lon": a["lon"], "n_piles": cfg["n_piles"], "power_kw": cfg["power_kw"],
                      "judet": a.get("judet"), "nume": f"Agent AI: {plan['zona']['nume']} #{a['pas']}"}
                     for a in plan["locatii_alese"]]
    return {
        "actiune": actiune, "parametri": p, "model": model_folosit,
        "raport": raport,
        "apeluri": [{k: v for k, v in c.items() if not k.startswith("_")} for c in calls],
        "harta": build_map(calls),
        "recomandari_salvabile": salvabile,
        "durata_s": round(time.time() - t0, 1),
    }


def status() -> dict:
    return {
        "configurat": bool(settings.gemini_api_key), "furnizor": "Google Gemini (google-genai)",
        "model": settings.gemini_model, "model_rezerva": settings.gemini_model_rezerva,
        "max_pasi": settings.agent_max_pasi,
        "actiuni": list(ACTIUNI), "tools": list(TOOLS),
    }


__all__ = ["run", "status", "LLMUnavailable", "ACTIUNI"]
