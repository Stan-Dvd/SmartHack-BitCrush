"""Agentul AI: rulează tool-urile deterministe pentru o acțiune din meniu și cere LLM-ului raportul.

Două moduri (AGENT_MOD în .env):
- "workflow" (implicit, eficient): pașii fiecărei acțiuni sunt ficși, deci tool-urile rulează direct în Python,
  iar LLM-ul primește UN SINGUR dosar compact cu rezultatele și scrie raportul -> un apel LLM pe acțiune,
  ~2-4k tokeni de intrare (față de ~25k în modul agentic, unde istoricul și schemele se retrimit la fiecare rundă).
- "agentic": LLM-ul alege singur tool-urile prin function calling (bucla din app/llm_*.py).

În ambele moduri toate cifrele vin din tool-uri, iar harta e construită DETERMINIST din rezultatele lor.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from . import llm
from .agent_tools import TOOLS, ToolError, for_llm, run_tool
from .config import settings
from .db import get_connection
from .llm import LLMUnavailable

ACTIUNI = ("sugereaza_statii", "evalueaza_punct", "compara_judete")

REGULI = """Reguli obligatorii:
- Toate cifrele din răspuns provin din DATE (rezultatele tool-urilor). Nu estima, nu inventa și nu recalcula
  valori; dacă o informație lipsește, spune că lipsește. Dacă DATE conțin o eroare, explic-o.
- Nu recomanda numărul de puncte (n_piles): efectul învățat nu e cauzal; folosește configurația din sarcină.
- Modelul nu prezice saturarea; nu face afirmații despre cozi sau stații pline.
- Când o locație are peste_baza = false, spune că e la nivelul de bază al modelului și că ordinea vine din
  criteriul secundar (populație, POI), nu dintr-o cerere prezisă mai mare.
- Limite de menționat pe scurt: model antrenat pe Shenzhen (valori absolute orientative, utile pentru
  comparații), pop_suspect (populație WorldPop nesigură), ipoteze economice ilustrative.
- Numere în format românesc: virgulă zecimală (938,9; 18,78), fără punct ca separator de mii (202509 sau
  202 509). kWh/zi prezis e un plafon (puncte ocupate x putere), nu energie livrată: energia livrată și banii
  vin din blocul economie.
- Markdown în română, concis (cel mult ~350 de cuvinte), cu secțiunile cerute; tabele pentru liste de locații;
  fără JSON brut."""

SYSTEM_WORKFLOW = ("Ești un analist de infrastructură de încărcare pentru vehicule electrice în România. "
                   "Primești rezultatele unor calcule deterministe (DATE, JSON compact) și scrii raportul.\n\n"
                   + REGULI)

SYSTEM_AGENTIC = ("Ești un analist de infrastructură de încărcare pentru vehicule electrice în România. "
                  "Folosești EXCLUSIV tool-urile disponibile; DATE = rezultatele lor. Pentru bani folosește "
                  "tool-ul economics.\n\n" + REGULI)

SECTIUNI = {
    "sugereaza_statii": "## Rezumat; ## Locații recomandate (tabel: #, coordonate/județ, ore încărcare/zi, câștig net "
                        "ore/zi, kWh/zi, peste nivelul de bază?); ## De ce aceste locații (top contribuții pe locație); "
                        "## Impact asupra stațiilor existente; ## Economie (ipoteze ilustrative); ## Limitări",
    "evalueaza_punct": "## Verdict (merită / nu merită, pe baza percentilelor); ## Cererea prezisă; ## De ce (top "
                       "contribuții); ## Canibalizare; ## Economie (ipoteze ilustrative); ## Limitări",
    "compara_judete": "## Rezumat; ## Clasament (tabel: județ, celule de oportunitate, ore max/zi, stații existente, "
                      "stații la 100k locuitori); ## Cel mai bun județ: locații de pornire; ## Cum citim clasamentul; "
                      "## Limitări",
}

ECON_KEYS = ("pret_kwh", "cost_energie_kwh", "capex_per_punct", "opex_procent_capex", "factor_putere_medie",
             "factor_ajustare_cerere")


# ============================================================================= utilitare
def _config(p: dict) -> tuple[int, float]:
    return (int(p.get("n_piles") or settings.predictie_n_piles_default),
            float(p.get("power_kw") or settings.predictie_power_kw_default))


def _econ(p: dict) -> dict:
    return {k: float(p[k]) for k in ECON_KEYS if p.get(k) not in (None, "")}


def _summary(result: Any, limit=280) -> str:
    s = json.dumps(for_llm(result), ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def _validate(actiune: str, p: dict):
    if actiune not in ACTIUNI:
        raise ValueError(f"Acțiune necunoscută: {actiune}. Disponibile: {', '.join(ACTIUNI)}")
    if actiune == "sugereaza_statii" and not p.get("zona"):
        raise ValueError("Parametrul „zona” e obligatoriu.")
    if actiune == "evalueaza_punct" and (p.get("lat") is None or p.get("lon") is None):
        raise ValueError("Parametrii lat și lon sunt obligatorii.")


# ============================================================================= extrase compacte pentru LLM
def _c_explain(r: dict) -> dict:
    comp = r.get("comparatie_judet") or {}
    return {
        "top3": [f"{e['descriere']}={e['valoare']} ({e['efect_ocupare']})" for e in r["top_3_contributii"]],
        "ocupare_baza_model": r["ocupare_de_baza_model"],
        "media_judet_ore_zi": comp.get("ore_incarcare_zi_medie_judet_statie_referinta"),
    }


def _c_econ(r: dict) -> dict:
    return {k: r[k] for k in ("kwh_livrati_an", "venit_an_ron", "profit_operational_an_ron", "capex_ron",
                              "ani_amortizare")}


# ============================================================================= fluxurile deterministe
Call = Callable[[str, dict], "dict | None"]


def _wf_sugereaza(p: dict, call: Call) -> dict:
    n, pw = _config(p)
    zona, judet = p["zona"], p.get("judet")
    k = int(p.get("k") or 3)
    data: dict[str, Any] = {"sarcina": {"zona": zona, "k": k, "n_piles": n, "power_kw": pw,
                                        "min_dist_intre_statii_m": float(p.get("min_dist_m", 1000)),
                                        "min_dist_fata_de_existente_m": float(p.get("min_dist_statie_m", 0))}}
    ra = call("resolve_area", {"nume": zona, "judet": judet})
    if not ra or not ra.get("gasit"):
        data["eroare_zona"] = for_llm(ra) if ra else "resolve_area a eșuat"
        return data
    data["zona"] = {k2: ra.get(k2) for k2 in ("nume", "tip", "judet", "populatie_ins", "suprafata_km2",
                                              "n_statii_existente")}
    zargs = {"zona": ra["nume"], "judet": ra["judet"], "tip": ra["tip"]}
    st = call("get_existing_stations", {**zargs, "top": 3})
    if st:
        data["statii_existente"] = {
            "n": st["n_statii"], "puncte": st.get("n_puncte"), "ore_zi_medie": st.get("ore_incarcare_zi_medie"),
            "ocupare_medie": st.get("ocupare_medie"),
            "operatori_top": [f"{o['operator']}:{o['statii']}" for o in st.get("pe_operator", [])[:3]],
            "cele_mai_utilizate": [f"{s['cod']} {s['localitate'] or ''}: {s['ore_incarcare_zi']} h/zi"
                                   for s in st.get("cele_mai_utilizate", [])],
        }
    plan = call("plan_network", {**zargs, "k": k, "n_piles": n, "power_kw": pw,
                                 "min_dist_m": float(p.get("min_dist_m", 1000)),
                                 "min_dist_statie_m": float(p.get("min_dist_statie_m", 0))})
    if not plan:
        data["eroare_plan"] = "plan_network a eșuat"
        return data
    econ_args = _econ(p)
    locatii, ipoteze = [], None
    for a in plan["locatii_alese"]:
        loc = {"nr": a["pas"], "lat": a["lat"], "lon": a["lon"], "judet": a["judet"],
               "ore_zi": a["ore_proprii_zi"], "castig_net_ore_zi": a["castig_net_ore_zi"],
               "pierdere_vecini_ore_zi": round(a["pierdere_statii_existente_ore_zi"]
                                               + a["pierdere_statii_propuse_ore_zi"], 3),
               "kwh_zi": a["kwh_zi_estimat"], "pop_1km": a["pop_1km"], "poi_500m": a["poi_500m"],
               "dist_statie_m": a["dist_nearest_station_m_initial"], "peste_baza": a["peste_nivelul_de_baza"]}
        ex = call("explain_cell", {"h3_index": a["h3_index"], "n_piles": n, "power_kw": pw})
        if ex:
            loc["explicatie"] = _c_explain(ex)
        ec = call("economics", {"kwh_zi": a["kwh_zi_estimat"], "n_piles": n, "power_kw": pw, **econ_args})
        if ec:
            loc["economie"] = _c_econ(ec)
            ipoteze = ec["ipoteze"]
        locatii.append(loc)
    data["plan"] = {"locatii": locatii, "total_ore_noi_zi": plan["total_ore_noi_zi"],
                    "total_castig_net_ore_zi": plan["total_castig_net_ore_zi"],
                    "nivel_baza_ore_zi": plan["nivel_de_baza_ore_zi"],
                    "statii_afectate": [f"{d['cod']}: {d['ore_inainte']}->{d['ore_dupa']} h/zi"
                                        for d in plan["statii_existente_afectate"][:5]]}
    if ipoteze:
        data["ipoteze_economice"] = ipoteze
    return data


def _wf_evalueaza(p: dict, call: Call) -> dict:
    n, pw = _config(p)
    lat, lon = float(p["lat"]), float(p["lon"])
    data: dict[str, Any] = {"sarcina": {"lat": lat, "lon": lon, "n_piles": n, "power_kw": pw}}
    sim = call("simulate_station", {"lat": lat, "lon": lon, "n_piles": n, "power_kw": pw})
    if not sim:
        data["eroare"] = "simulate_station a eșuat"
        return data
    s = sim["sumar"]
    data["predictie"] = {"judet": sim["punct"]["judet"], "ore_zi": s["ore_incarcare_zi"], "kwh_zi": s["kwh_zi"],
                         "ocupare_medie": s["ocupare_medie"], "ocupare_varf": s["ocupare_varf"],
                         "ora_varf": s["ora_varf"],
                         "percentila_vs_statii_existente": sim["percentila_ocupare_vs_statii_existente"],
                         "percentila_vs_grila_h3": sim["percentila_vs_grila_h3"],
                         "pop_suspect": sim["pop_suspect"], "context": sim["motiv"]}
    can = sim["canibalizare"]
    data["canibalizare"] = {"statii_pe_3km": can["statii_pe_3km"], "pierdere_totala_ore_zi": can["pierdere_totala_ore_zi"],
                            "cele_mai_afectate": [f"{c['cod']} ({c['dist_m']} m): {c['delta']} h/zi"
                                                  for c in can["cele_mai_afectate"][:3] if c["delta"] < 0]}
    ex = call("explain_cell", {"lat": lat, "lon": lon, "n_piles": n, "power_kw": pw})
    if ex:
        data["explicatie"] = _c_explain(ex)
    ec = call("economics", {"kwh_zi": s["kwh_zi"], "n_piles": n, "power_kw": pw, **_econ(p)})
    if ec:
        data["economie"] = _c_econ(ec)
        data["ipoteze_economice"] = ec["ipoteze"]
    return data


def _wf_compara(p: dict, call: Call) -> dict:
    n, pw = _config(p)
    data: dict[str, Any] = {"sarcina": {"n_piles": n, "power_kw": pw}}
    ra = call("rank_areas", {"top": 10, "n_piles": n, "power_kw": pw})
    if not ra:
        data["eroare"] = "rank_areas a eșuat"
        return data
    data["definitie_celula_oportunitate"] = ra["definitie"]
    data["clasament"] = [
        {"loc": r["loc"], "judet": r["judet"], "celule": r["celule_oportunitate"], "ore_max": r["ore_max_zi"],
         "statii": r["statii_existente"], "la_100k": r["statii_la_100k_loc"],
         **({"pop_suspect": True} if r["populatie_suspecta"] else {})}
        for r in ra["clasament"]]
    if ra["clasament"]:
        primul = ra["clasament"][0]["judet"]
        sc = call("score_candidates", {"zona": primul, "tip": "judet", "top_k": 3, "n_piles": n, "power_kw": pw})
        if sc:
            data["primul_judet"] = {"judet": primul, "candidati": [
                {"lat": c["lat"], "lon": c["lon"], "ore_zi": c["ore_incarcare_zi"], "pop_1km": c["pop_1km"],
                 "poi_500m": c["poi_500m"], "dist_statie_m": c["dist_nearest_station_m"],
                 "peste_baza": c["peste_nivelul_de_baza"]} for c in sc["candidati"]]}
    return data


WORKFLOWS = {"sugereaza_statii": _wf_sugereaza, "evalueaza_punct": _wf_evalueaza, "compara_judete": _wf_compara}


def _writer_prompt(actiune: str, data: dict) -> str:
    return (f"Sarcina: raport pentru acțiunea „{actiune}”.\nSecțiuni: {SECTIUNI[actiune]}.\n\n"
            f"DATE:\n{_compact_json(data)}")


# ============================================================================= modul agentic (opțional)
def build_prompt(actiune: str, p: dict) -> str:
    n, pw = _config(p)
    econ = _compact_json(_econ(p)) if _econ(p) else "valorile implicite ale tool-ului"
    if actiune == "sugereaza_statii":
        k = int(p.get("k") or 3)
        judet = f", judet=„{p['judet']}”" if p.get("judet") else ""
        steps = (f"1. resolve_area; 2. get_existing_stations; 3. plan_network cu k={k}, n_piles={n}, power_kw={pw}, "
                 f"min_dist_m={p.get('min_dist_m', 1000)}, min_dist_statie_m={p.get('min_dist_statie_m', 0)}; "
                 f"4. explain_cell pentru fiecare locație aleasă; 5. economics pentru fiecare, cu kwh_zi = "
                 f"kwh_zi_estimat și ipotezele {econ}.")
        task = f"Propune {k} locații noi în zona „{p['zona']}”{judet}."
    elif actiune == "evalueaza_punct":
        steps = (f"1. simulate_station (lat={p['lat']}, lon={p['lon']}, n_piles={n}, power_kw={pw}); 2. explain_cell; "
                 f"3. economics cu kwh_zi = sumar.kwh_zi și ipotezele {econ}.")
        task = f"Evaluează o stație nouă în punctul lat={p['lat']}, lon={p['lon']}."
    else:
        steps = f"1. rank_areas cu top=10, n_piles={n}, power_kw={pw}; 2. score_candidates pentru primul județ (top_k=3)."
        task = "Unde în România merită investit în stații noi?"
    return f"Sarcină: {task}\nPași: {steps}\nSecțiuni în raport: {SECTIUNI[actiune]}."


# ============================================================================= harta
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


# ============================================================================= rularea unei acțiuni
def run(actiune: str, parametri: dict | None = None) -> dict:
    """Rulează o acțiune din meniu. Ridică LLMUnavailable sau ValueError (parametri greșiți)."""
    p = dict(parametri or {})
    _validate(actiune, p)
    if not llm.is_configured():  # nu rulăm calculele dacă oricum nu putem scrie raportul
        llm.check_configured()
    calls: list[dict] = []
    t0 = time.time()

    def record(name: str, args: dict) -> tuple[dict | None, dict]:
        t = time.time()
        entry = {"tool": name, "argumente": {k: v for k, v in args.items() if v is not None}, "ok": True}
        try:
            with get_connection() as conn:
                result = run_tool(conn, name, args)
            entry.update(_result=result, rezumat=_summary(result))
            payload = for_llm(result)
        except ToolError as e:
            result = None
            entry.update(ok=False, _result=None, rezumat=f"Eroare: {e}")
            payload = {"eroare": str(e)}
        except Exception as e:  # eroare neașteptată într-un tool: apare în raport, nu oprește analiza
            result = None
            entry.update(ok=False, _result=None, rezumat=f"Eroare internă: {type(e).__name__}: {e}")
            payload = {"eroare": f"Eroare internă în tool-ul {name}."}
        entry["durata_s"] = round(time.time() - t, 2)
        calls.append(entry)
        return result, payload

    modele = llm.models()
    raport, model_folosit, usage = None, None, None
    mod = (settings.agent_mod or "workflow").lower()
    if mod == "agentic":
        for i, model in enumerate(modele):
            calls.clear()
            try:
                raport = llm.run_tool_loop(SYSTEM_AGENTIC, build_prompt(actiune, p), TOOLS,
                                           lambda n_, a_: record(n_, a_)[1], model=model)
                model_folosit = model
                break
            except LLMUnavailable as e:
                if not e.temporar or i == len(modele) - 1:
                    raise
    else:
        data = WORKFLOWS[actiune](p, lambda n_, a_: record(n_, a_)[0])
        prompt = _writer_prompt(actiune, data)
        for i, model in enumerate(modele):
            try:
                raport, usage = llm.generate_text(SYSTEM_WORKFLOW, prompt, model=model)
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
        "actiune": actiune, "parametri": p, "furnizor": llm.provider(), "model": model_folosit, "mod": mod,
        "raport": raport,
        "tokeni": usage,
        "apeluri": [{k: v for k, v in c.items() if not k.startswith("_")} for c in calls],
        "harta": build_map(calls),
        "recomandari_salvabile": salvabile,
        "durata_s": round(time.time() - t0, 1),
    }


def status() -> dict:
    return {
        "configurat": llm.is_configured(), "furnizor": llm.provider_label(), "modele": llm.models(),
        "mod": settings.agent_mod, "max_tokeni_raport": settings.agent_max_output_tokens,
        "actiuni": list(ACTIUNI), "tools": list(TOOLS),
    }


__all__ = ["run", "status", "LLMUnavailable", "ACTIUNI"]
