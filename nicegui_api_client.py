"""Client HTTP pentru frontendul NiceGUI existent."""
import os
import httpx

API_URL = os.getenv("URBANEV_API_URL", "http://127.0.0.1:8000")

def _get(path, params=None):
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def load_all_stations():
    rows = _get("/api/stations", {"limit": 20000})
    return [{
        **r,
        "id": f"RO-EV-{r['statie_id']}",
        "name": f"Stație EV {r['statie_id']}",
        "lat": r["lat"],
        "lon": r["lng"],
        "localitate": f"București - zonă TAZ {r['taz_id']}" if r.get("taz_id") is not None else "București",
        "judet": "București",
        "operator": "UrbanEV",
        "power_max_kw": float(os.getenv("PUMP_RATING_KW", "50")),
        "power_count": r.get("charge_count", 0),
        "capacity": r.get("charge_count", 0),
        "h3_res8": r["h3_res8"],
        "h3_res9": r["h3_res9"],
        "is_open_24_7": 1,
        "synthetic": int(r["statie_id"]) >= 900000,
    } for r in rows]

def get_demand_points(start=None, end=None, resolution=9):
    params = {"resolution": resolution}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return _get("/api/heatmap", params)

def get_csv_stats(stations):
    return {
        "avg_power": round(sum(float(s.get("power_max_kw") or 0) for s in stations) / len(stations), 1) if stations else 0,
        "open_24": sum(int(s.get("is_open_24_7") or 0) for s in stations),
        "counties": len({s.get("judet") for s in stations if s.get("judet")}),
    }

def get_top_ai_insights():
    rows = _get("/api/insights", {"limit": 3, "resolution": 8})
    return [{
        "priority": f"PRIORITATE {i+1}",
        "type": "H3_DEMAND",
        "title": f"Celula H3 {r['h3_index']}",
        "score": r.get("total_volume_kwh"),
        "reason": (
            f"Stații: {r['station_count']}; pompe ocupate în medie: {r.get('avg_busy')}; "
            f"energie estimată: {r.get('total_volume_kwh')} kWh"
        ),
    } for i, r in enumerate(rows)]
