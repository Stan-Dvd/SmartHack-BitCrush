"""Client HTTP pentru frontendul NiceGUI: stațiile, heatmap-ul și insight-urile vin din predicțiile modelului ML."""
import os

import httpx

API_URL = os.getenv("URBANEV_API_URL", "http://127.0.0.1:8000")


def _get(path, params=None):
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _post(path, payload):
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def load_all_stations():
    """Stațiile din PostgreSQL, în schema folosită de main.py (aceleași chei ca CSV-ul de rezervă)."""
    rows = _get("/api/stations", {"limit": 20000})
    return [{
        **r,
        "id": r["cod"],
        "name": f"Stație {r['operator'] or 'EV'} {r['localitate'] or ''}".strip(),
        "lat": r["lat"],
        "lon": r["lng"],
        "localitate": r.get("localitate") or "Fără Localitate",
        "judet": r.get("judet") or "Fără Județ",
        "operator": r.get("operator") or "Necunoscut / Independent",
        "power_max_kw": float(r.get("power_max_kw") or r.get("power_kw") or 0),
        "power_count": r.get("n_piles", 0),
        "capacity": r.get("n_piles", 0),
        "is_open_24_7": int(bool(r.get("is_open_24_7"))),
        "h3_res8": r["h3_res8"],
        "h3_res9": r["h3_res9"],
        "synthetic": False,
        # predicția modelului activ (ocupare medie și ore de încărcare pe zi)
        "ore_incarcare_zi_pred": r.get("ore_incarcare_zi"),
        "ocupare_medie_pred": r.get("ocupare_medie"),
    } for r in rows]


def get_demand_points(start=None, end=None, resolution=8, sursa="predictie"):
    """Puncte {lat, lon, score} pentru heatmap. Implicit: utilizarea prezisă a unei stații noi de referință
    în fiecare celulă H3 populată (harta de oportunități)."""
    params = {"resolution": resolution, "sursa": sursa}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return _get("/api/heatmap", params)


def get_station_forecast(statie_id, start=None, end=None, calibrat=True):
    params = {"calibrat": calibrat}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return _get(f"/api/stations/{statie_id}/forecast", params)


def evaluate_location(lat, lon, n_piles=None, power_kw=None, cu_impact=True, salveaza=False):
    """Performanța prezisă a unei stații noi în (lat, lon) și impactul asupra stațiilor vecine."""
    return _post("/api/candidates/evaluate", {
        "latitude": lat, "longitude": lon, "n_piles": n_piles, "power_kw": power_kw,
        "cu_impact": cu_impact, "salveaza": salveaza,
    })


class AgentError(RuntimeError):
    """Eroare afișabilă utilizatorului (agent neconfigurat, limită Gemini, parametri greșiți)."""


def _detail(response):
    try:
        return response.json().get("detail") or response.text
    except ValueError:
        return response.text


def agent_status():
    return _get("/api/agent/status")


def run_agent(actiune, parametri, timeout=300):
    """Rulează o acțiune a agentului AI: {raport, apeluri, harta, recomandari_salvabile}."""
    try:
        r = httpx.post(f"{API_URL}/api/agent/run", json={"actiune": actiune, "parametri": parametri}, timeout=timeout)
    except httpx.HTTPError as e:
        raise AgentError(f"API-ul nu răspunde ({API_URL}): {e}") from e
    if r.status_code >= 400:
        raise AgentError(str(_detail(r)))
    return r.json()


def save_candidates(candidati, status="propus"):
    """candidati: [{lat, lon, n_piles, power_kw, nume, judet}] -> locatie_candidat."""
    payload = {"status": status, "candidati": [
        {"latitude": c["lat"], "longitude": c["lon"], "n_piles": c.get("n_piles"), "power_kw": c.get("power_kw"),
         "nume": c.get("nume"), "judet": c.get("judet")} for c in candidati]}
    r = httpx.post(f"{API_URL}/api/candidates/batch", json=payload, timeout=120)
    if r.status_code >= 400:
        raise AgentError(str(_detail(r)))
    return r.json()


def get_csv_stats(stations):
    return {
        "avg_power": round(sum(float(s.get("power_max_kw") or 0) for s in stations) / len(stations), 1) if stations else 0,
        "open_24": sum(int(s.get("is_open_24_7") or 0) for s in stations),
        "counties": len({s.get("judet") for s in stations if s.get("judet")}),
    }


def get_top_ai_insights(limit=3):
    """Top locații noi (celule H3 la cel puțin 1 km de o stație existentă), după utilizarea prezisă."""
    rows = _get("/api/insights", {"limit": limit})
    return [{
        "priority": f"PRIORITATE {i + 1}",
        "type": r["type"],
        "title": r["title"],
        "score": f"{r['score']:.1f} h/zi",
        "reason": r["reason"],
        "lat": r["lat"],
        "lon": r["lon"],
        "localitate": r.get("localitate") or r.get("judet"),
        "judet": r.get("judet"),
    } for i, r in enumerate(rows)]
