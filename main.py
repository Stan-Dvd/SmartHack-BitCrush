from nicegui import ui
import pandas as pd
import json

def _log(message):
    """print() care nu poate opri aplicația din cauza codificării consolei (ex. cp1252 pe Windows)."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "backslashreplace").decode("ascii"))


def load_all_stations():
    # Încearcă mai întâi baza de date; CSV-ul rămâne fallback dacă API-ul nu este pornit.
    try:
        from nicegui_api_client import load_all_stations as load_from_api
        rows = load_from_api()
        if rows:
            _log(f"UrbanEV: {len(rows)} stații încărcate din PostgreSQL.")
            return rows
    except Exception as e:
        _log(f"API indisponibil, încerc citirea CSV: {e}")

    try:
        df = pd.read_csv('statii_electrice_ML.csv')
        for column, default in {
            'operator': 'Necunoscut / Independent',
            'judet': 'Fără Județ',
            'localitate': 'Fără Localitate',
            'power_max_kw': 0,
            'power_count': 0,
            'capacity': 0,
            'is_open_24_7': 0,
        }.items():
            if column not in df.columns:
                df[column] = default
            df[column] = df[column].fillna(default)
        return df.to_dict(orient='records')
    except Exception as e:
        _log(f"Eroare la citirea CSV-ului: {e}")
        return []
    
ALL_RO_STATIONS = load_all_stations()


#   COORDONATE JUDETE ROMANIA       
ROMANIAN_COUNTIES = {
    'România (Toată Țara)': (45.9432, 24.9668, 7),
    'Alba': (46.0686, 23.5714, 9),
    'Arad': (46.1866, 21.3123, 9),
    'Argeș': (44.8565, 24.8692, 9),
    'Bacău': (46.5670, 26.9146, 9),
    'Bihor': (47.0500, 21.9333, 9),
    'Bistrița-Năsăud': (47.1333, 24.5000, 9),
    'Botoșani': (47.7447, 26.6631, 9),
    'Brașov': (45.6580, 25.6012, 10),
    'Brăila': (45.2692, 27.9575, 9),
    'București': (44.4323, 26.1063, 11),
    'Buzău': (45.1500, 26.8333, 9),
    'Caraș-Severin': (45.2000, 21.9000, 9),
    'Călărași': (44.2000, 27.3333, 9),
    'Cluj': (46.7712, 23.6236, 10),
    'Constanța': (44.1792, 28.6498, 9),
    'Covasna': (45.8667, 25.7833, 9),
    'Dâmbovița': (44.9333, 25.4500, 9),
    'Dolj': (44.3333, 23.8000, 9),
    'Galați': (45.4333, 28.0500, 9),
    'Giurgiu': (43.9000, 25.9667, 9),
    'Gorj': (45.0333, 23.2833, 9),
    'Harghita': (46.3667, 25.8000, 9),
    'Hunedoara': (45.8833, 22.9000, 9),
    'Ialomița': (44.5667, 27.3667, 9),
    'Iași': (47.1585, 27.6014, 9),
    'Ilfov': (44.5333, 26.0833, 10),
    'Maramureș': (47.6500, 23.5833, 9),
    'Mehedinți': (44.6333, 22.6500, 9),
    'Mureș': (46.5456, 24.5625, 9),
    'Neamț': (46.9275, 26.3708, 9),
    'Olt': (44.4333, 24.3667, 9),
    'Prahova': (45.1000, 26.0167, 9),
    'Satu Mare': (47.7900, 22.8900, 9),
    'Sălaj': (47.1833, 23.0500, 9),
    'Sibiu': (45.7983, 24.1257, 10),
    'Suceava': (47.6514, 26.2556, 9),
    'Teleorman': (43.9667, 25.3333, 9),
    'Timiș': (45.7537, 21.2257, 9),
    'Tulcea': (45.1789, 28.8003, 9),
    'Vaslui': (46.6333, 27.7333, 9),
    'Vâlcea': (45.1000, 24.3667, 9),
    'Vrancea': (45.7000, 27.1833, 9),
}


# =====================================================================
# DATE STAȚII PPC — ÎNCĂRCARE DINAMICĂ
# =====================================================================
# Ordinea surselor (prima care returnează stații valide câștigă):
#   1. Fișier dedicat (JSON sau CSV) — calea din env PPC_STATIONS_FILE,
#      implicit `ppc_stations.json` lângă main.py.
#   2. Filtrare din ALL_RO_STATIONS (API/PostgreSQL sau CSV) după operator PPC / EvPass.
# Nu există date hardcodate: dacă nicio sursă nu are stații PPC, PPC_STATIONS = [].
# Indiferent de sursă, fiecare stație are exact cheile din PPC_STATION_KEYS.

PPC_STATION_KEYS = (
    "id", "name", "city", "county", "lat", "lng",
    "type", "ports", "ai_action", "ai_reason", "utilization",
)

# Operatorii considerați „PPC” la filtrarea setului național (comparație case-insensitive)
PPC_OPERATOR_PATTERN = r"\bppc\b|ev\s*pass|ppc\s*blue"

def _to_float(value):
    """Convertește la float; întoarce None pentru valori lipsă / NaN / invalide."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _normalize_ppc_station(raw, index):
    """Aduce o înregistrare (din fișier dedicat) la schema PPC_STATION_KEYS.
    Acceptă și `lon` în loc de `lng`. Întoarce None dacă lipsesc coordonatele valide."""
    lat = _to_float(raw.get("lat"))
    lng = _to_float(raw.get("lng", raw.get("lon")))
    if lat is None or lng is None:
        return None
    defaults = {
        "id": f"PPC-{index + 1:03d}",
        "name": "Stație PPC",
        "city": "-",
        "county": "-",
        "type": "N/A",
        "ports": "N/A",
        "ai_action": "PPC",
        "ai_reason": "Fără analize suplimentare.",
        "utilization": "N/A",
    }
    station = {}
    for key in PPC_STATION_KEYS:
        if key == "lat":
            station[key] = lat
        elif key == "lng":
            station[key] = lng
        else:
            value = raw.get(key)
            station[key] = defaults[key] if value is None or (isinstance(value, float) and value != value) else str(value)
    return station


def _utilization_text(st, fallback="N/A"):
    """Gradul de utilizare prezis de modelul ML (din API); fallback dacă stația nu are predicție."""
    occ = _to_float(st.get("ocupare_medie_pred"))
    if occ is None:
        return fallback
    hours = _to_float(st.get("ore_incarcare_zi_pred")) or 0
    return f"{occ:.0%} ocupare prezisă ({hours:.1f} h încărcare/zi, model ML)"


def _ppc_from_national_record(st, index):
    """Mapează o stație din ALL_RO_STATIONS (schema API/CSV) la schema PPC."""
    lat, lng = _to_float(st.get("lat")), _to_float(st.get("lon", st.get("lng")))
    if lat is None or lng is None:
        return None
    power = _to_float(st.get("power_max_kw")) or 0
    count = int(_to_float(st.get("power_count")) or 0)
    if power >= 150:
        st_type = "DC Ultra-Fast"
    elif power >= 50:
        st_type = "DC Fast"
    elif power > 22:
        st_type = "AC / DC"
    else:
        st_type = "AC Standard"
    city = st.get("localitate") or "-"
    county = st.get("judet") or "-"
    return {
        "id": f"PPC-{index + 1:03d}",
        "name": st.get("name") or f"PPC {city}",
        "city": city,
        "county": county,
        "lat": lat,
        "lng": lng,
        "type": st_type,
        "ports": f"Putere Max: {power:g} kW (Porturi: {count})",
        "ai_action": "PPC",
        "ai_reason": f"Stație din rețeaua {st.get('operator', 'PPC')}. "
                     f"Program 24/7: {'Da' if st.get('is_open_24_7') == 1 else 'Nu/Necunoscut'}.",
        "utilization": _utilization_text(st),
    }


def _load_ppc_from_file():
    import os
    from pathlib import Path
    path = Path(os.getenv("PPC_STATIONS_FILE", Path(__file__).with_name("ppc_stations.json")))
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        records = pd.read_csv(path).to_dict(orient="records")
    else:
        with open(path, encoding="utf-8") as fh:
            records = json.load(fh)
        if isinstance(records, dict):  # acceptă și {"stations": [...]}
            records = records.get("stations", [])
    stations = [_normalize_ppc_station(r, i) for i, r in enumerate(records) if isinstance(r, dict)]
    return [s for s in stations if s]


def _load_ppc_from_national(stations):
    import re
    pattern = re.compile(PPC_OPERATOR_PATTERN, re.IGNORECASE)
    matches = [st for st in stations if pattern.search(str(st.get("operator") or ""))]
    result = [_ppc_from_national_record(st, i) for i, st in enumerate(matches)]
    return [s for s in result if s]


def load_ppc_stations():
    """Încarcă stațiile PPC doar din surse dinamice; întoarce [] dacă nu găsește nimic.
    Nu ridică niciodată excepții."""
    sources = (
        ("fișier dedicat", _load_ppc_from_file),
        ("setul național (operator PPC/EvPass)", lambda: _load_ppc_from_national(ALL_RO_STATIONS)),
    )
    for label, loader in sources:
        try:
            stations = loader()
            if stations:
                _log(f"UrbanEV: {len(stations)} stații PPC încărcate din {label}.")
                return stations
        except Exception as e:
            _log(f"Stații PPC: sursa '{label}' indisponibilă: {e}")
    _log("UrbanEV: nicio stație PPC găsită în sursele dinamice (ppc_stations.json / API).")
    return []


PPC_STATIONS = load_ppc_stations()

 
# =====================================================================
# HEATMAP CERERE - CONTRACT DE DATE PENTRU MODELUL ML
# =====================================================================
# Modelul ML va trebui să returneze o listă de puncte de forma:
#     [{'lat': 44.43, 'lon': 26.10, 'score': 0.87}, ...]
# unde 'score' = cererea prezisă (orice scară; o normalizăm aici în [0, 1]).
# Pentru a conecta ML-ul, înlocuiești DOAR corpul funcției get_demand_points().

# Hub-uri cu cerere mare (MOCK): (lat, lon, pondere, nr. puncte, împrăștiere în grade)
_MOCK_HOTSPOTS = [
    (44.4323, 26.1063, 1.00, 60, 0.10),  # București
    (46.7712, 23.6236, 0.85, 35, 0.06),  # Cluj-Napoca
    (45.7537, 21.2257, 0.80, 30, 0.06),  # Timișoara
    (47.1585, 27.6014, 0.70, 25, 0.06),  # Iași
    (45.6580, 25.6012, 0.75, 25, 0.05),  # Brașov
    (44.1792, 28.6498, 0.70, 25, 0.06),  # Constanța
    (45.7983, 24.1257, 0.65, 20, 0.05),  # Sibiu
    (44.9461, 26.0130, 0.55, 15, 0.05),  # Ploiești
    (44.8565, 24.8692, 0.50, 15, 0.05),  # Pitești
    (44.3302, 23.7949, 0.50, 15, 0.05),  # Craiova
]

# Coridoare autostradă (MOCK): cerere de tranzit între două capete
_MOCK_CORRIDORS = [
    ((44.4323, 26.1063), (44.8565, 24.8692), 0.60),  # A1 București - Pitești
    ((44.4323, 26.1063), (44.1792, 28.6498), 0.55),  # A2 București - Constanța
    ((44.4323, 26.1063), (44.9461, 26.0130), 0.55),  # A3 București - Ploiești
    ((45.7983, 24.1257), (45.7537, 21.2257), 0.45),  # A1 Sibiu - Timișoara
    ((46.7712, 23.6236), (46.5456, 24.5625), 0.40),  # A3 Cluj - Târgu Mureș
]

def get_csv_stats(df_dict):
    if not df_dict:
        return {"avg_power": 0, "open_24": 0, "counties": 0}
    
    df = pd.DataFrame(df_dict)
    avg_power = round(df['power_max_kw'].mean(), 1)
    open_24_count = int((df['is_open_24_7'] == 1).sum())
    counties_count = int(df['judet'].nunique())
    
    return {
        "avg_power": avg_power,
        "open_24": open_24_count,
        "counties": counties_count
    }

STATS = get_csv_stats(ALL_RO_STATIONS)

def get_demand_points():
    """Citește agregările H3 din PostgreSQL; păstrează mock-ul ca fallback."""
    try:
        from nicegui_api_client import get_demand_points as get_api_demand_points
        points = get_api_demand_points(resolution=9)
        if points:
            return points
    except Exception as e:
        print(f"Heatmap API indisponibil, folosesc date mock: {e}")

    import numpy as np
    rng = np.random.default_rng(42)
    points = []
    for lat, lon, weight, n, spread in _MOCK_HOTSPOTS:
        for _ in range(n):
            dlat, dlon = rng.normal(0, spread, 2)
            distance = (dlat ** 2 + dlon ** 2) ** 0.5
            score = weight * np.exp(-distance / (2 * spread)) * rng.uniform(0.7, 1.0)
            points.append({'lat': lat + dlat, 'lon': lon + dlon, 'score': score})
    for (lat1, lon1), (lat2, lon2), weight in _MOCK_CORRIDORS:
        for t in np.linspace(0, 1, 25):
            points.append({
                'lat': lat1 + (lat2 - lat1) * t + rng.normal(0, 0.01),
                'lon': lon1 + (lon2 - lon1) * t + rng.normal(0, 0.01),
                'score': weight * rng.uniform(0.6, 1.0),
            })
    for st in ALL_RO_STATIONS:
        points.append({
            'lat': st.get('lat'),
            'lon': st.get('lon'),
            'score': 0.08 + 0.17 * min(float(st.get('power_max_kw') or 0) / 150.0, 1.0),
        })
    return points


def to_heat_data(points):
    """Curăță + normalizează punctele în formatul leaflet.heat: [[lat, lon, intensitate 0..1], ...]."""
    clean = []
    for p in points:
        try:
            lat, lon, score = float(p['lat']), float(p['lon']), float(p['score'])
        except (KeyError, TypeError, ValueError):
            continue  # punct invalid -> ignorat
        # Elimină NaN și coordonatele din afara României (aprox.)
        if any(v != v for v in (lat, lon, score)):
            continue
        if not (43.5 <= lat <= 48.5 and 20.0 <= lon <= 30.0):
            continue
        clean.append((lat, lon, score))

    if not clean:
        return []
    max_score = max(s for _, _, s in clean) or 1.0
    return [[round(lat, 5), round(lon, 5), round(max(s, 0) / max_score, 3)] for lat, lon, s in clean]


# =====================================================================
# AI DECISION INSIGHTS - TOP OPORTUNITĂȚI (PREGĂTIT PENTRU ML)
# =====================================================================
def get_top_ai_insights():
    """Construiește lista de oportunități din agregările H3 stocate în PostgreSQL."""
    try:
        from nicegui_api_client import get_top_ai_insights as get_api_insights
        rows = get_api_insights()
        if rows:
            return [
                {
                    "priority": f"PRIORITATE {i + 1}",
                    "type": row.get("type", "NEW_SITE"),
                    "title": row.get("title", ""),
                    "score": row.get("score"),
                    "reason": row.get("reason", ""),
                    "lat": row.get("lat"),
                    "lon": row.get("lon"),
                    "localitate": row.get("localitate"),
                    "border_color": "border-teal-500/40",
                    "text_color": "text-teal-400 font-bold",
                    "badge_color": "teal-900",
                    "badge_text_color": "text-teal-300",
                }
                for i, row in enumerate(rows)
            ]
    except Exception as e:
        print(f"Insights API indisponibil, folosesc exemple mock: {e}")

    return [
        {
            "priority": "PRIORITATE 1",
            "type": "NEW_SITE",
            "title": "București - zonă cu cerere ridicată",
            "border_color": "border-teal-500/40",
            "text_color": "text-teal-400 font-bold",
            "badge_color": "teal-900",
            "badge_text_color": "text-teal-300"
        },
        {
            "priority": "PRIORITATE 2",
            "type": "UPSCALE",
            "title": "București - capacitate de extins",
            "border_color": "border-amber-500/40",
            "text_color": "text-amber-400 font-bold",
            "badge_color": "amber-900",
            "badge_text_color": "text-amber-300"
        },
        {
            "priority": "PRIORITATE 3",
            "type": "NEW_SITE",
            "title": "București - oportunitate H3",
            "border_color": "border-sky-500/40",
            "text_color": "text-sky-400 font-bold",
            "badge_color": "sky-900",
            "badge_text_color": "text-sky-300"
        }
    ]


# def get_top_ai_insights():
#     # Când vin datele reale de la ML:
#     df_ml = pd.read_csv('rezultate_ml.csv') # Coloane: locatie, scor, actiune_recomandata, motiv
#     top3 = df_ml.sort_values(by='scor', ascending=False).head(3)
    
#     results = []
#     colors = [
#         ("border-teal-500/40", "text-teal-400 font-bold", "teal-900", "text-teal-300"),
#         ("border-amber-500/40", "text-amber-400 font-bold", "amber-900", "text-amber-300"),
#         ("border-sky-500/40", "text-sky-400 font-bold", "sky-900", "text-sky-300"),
#     ]
    
#     for i, (_, row) in enumerate(top3.iterrows()):
#         b_col, t_col, bg_badge, t_badge = colors[i]
#         results.append({
#             "priority": f"PRIORITATE {i+1}",
#             "type": row['actiune_recomandata'],
#             "title": row['locatie'],
#             "score": f"{int(row['scor'] * 100)}%",
#             "reason": row['motiv'],
#             "border_color": b_col,
#             "text_color": t_col,
#             "badge_color": bg_badge,
#             "badge_text_color": t_badge
#         })
#     return results



HEATMAP_OPTIONS = {
    'radius': 22,
    'blur': 18,
    'maxZoom': 7,        # la zoom >= 7 intensitatea e afișată 100% (înainte era 10 => totul de 8x mai slab)
    'max': 1.5,          # puncte suprapuse se adună; >1 evită ca totul să devină roșu
    'minOpacity': 0.05,  # zonele cu cerere mică rămân aproape transparente => se vede harta
    'gradient': {'0.2': 'blue', '0.45': 'cyan', '0.65': 'lime', '0.85': 'yellow', '1.0': 'red'},
}

# Transparența întregului strat heatmap (0 = invizibil, 1 = opac). 0.6 => harta se vede prin culori.
HEATMAP_LAYER_OPACITY = 0.6

heat_state = {'layer': None}


# INTERFATA GRAFICA

all_ro_markers = []

ui.page_title('PPC EV Network AI Agent')
# Pluginurile Leaflet (heat, markercluster) NU se mai încarcă aici în <head>:
# NiceGUI încarcă Leaflet abia când se creează harta, deci în <head> `L` nu există încă
# și pluginul crăpa. Le dăm direct hărții prin `additional_resources` (vezi ui.leaflet mai jos).
LEAFLET_PLUGINS = [
    'https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js',
    'https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.css',
    'https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css',
    'https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js',
]

# Heatmap semi-transparent: canvas-ul pluginului primește clasa 'leaflet-heatmap-layer'
ui.add_head_html(f'''
<style>
    .leaflet-heatmap-layer {{ opacity: {HEATMAP_LAYER_OPACITY}; }}
</style>
''')

# MARKER CLUSTERING — stil Dark Slate / Teal (PPC Theme)
# Notă: div-ul exterior (.ppc-cluster) e poziționat de Leaflet prin `transform`,
# deci efectul de hover (scale) se aplică pe div-ul interior, ca să nu strice poziția.
ui.add_head_html('''
<style>
    .ppc-cluster { background: transparent; border: none; }
    .ppc-cluster > div {
        width: 100%; height: 100%;
        display: flex; align-items: center; justify-content: center;
        box-sizing: border-box;
        border-radius: 50%;
        background: rgba(15, 23, 42, 0.95);              /* slate-900 */
        border: 2px solid #2dd4bf;                       /* teal-400 */
        color: #2dd4bf;
        font-weight: 800; font-family: ui-monospace, monospace;
        box-shadow: 0 0 6px rgba(45, 212, 191, 0.55),
                    0 0 14px rgba(45, 212, 191, 0.25),
                    inset 0 0 6px rgba(45, 212, 191, 0.15);
        transition: transform .18s ease, border-color .18s ease,
                    box-shadow .18s ease, color .18s ease;
        cursor: pointer;
    }
    .ppc-cluster-sm > div { font-size: 12px; }
    .ppc-cluster-md > div { font-size: 13px; border-width: 2.5px; }
    .ppc-cluster-lg > div { font-size: 14px; border-width: 3px; }
    .ppc-cluster:hover > div {
        transform: scale(1.15);
        border-color: #5eead4;                           /* teal-300 */
        color: #ccfbf1;
        box-shadow: 0 0 10px rgba(94, 234, 212, 0.8),
                    0 0 22px rgba(45, 212, 191, 0.45),
                    inset 0 0 8px rgba(94, 234, 212, 0.25);
    }
    /* ---------- VARIANTA PPC (stații proprii): Electric Cyan ---------- */
    /* Clusterele și pinii PPC stau mereu deasupra celor din CSV (Leaflet pune z-index inline) */
    .ppc-cluster.ppc-own, .ppc-pin { z-index: 10000 !important; }
    .ppc-cluster.ppc-own > div {
        border-color: #00e5ff;                           /* electric cyan */
        color: #7df3ff;                                  /* cyan deschis */
        box-shadow: 0 0 6px rgba(0, 229, 255, 0.6),
                    0 0 16px rgba(0, 170, 255, 0.35),
                    inset 0 0 6px rgba(0, 229, 255, 0.18);
    }
    .ppc-cluster.ppc-own:hover > div {
        border-color: #ccf9ff;                           /* cyan pal */
        color: #effeff;
        box-shadow: 0 0 10px rgba(125, 243, 255, 0.9),
                    0 0 26px rgba(0, 170, 255, 0.55),
                    inset 0 0 8px rgba(125, 243, 255, 0.3);
    }

    /* Pin individual PPC (SVG în L.divIcon) */
    .ppc-pin { background: transparent; border: none; }
    .ppc-pin svg {
        width: 100%; height: 100%; overflow: visible;
        filter: drop-shadow(0 0 4px rgba(0, 229, 255, 0.85))
                drop-shadow(0 0 10px rgba(0, 170, 255, 0.45));
        transform-origin: 50% 100%;
        transition: transform .18s ease, filter .18s ease;
        cursor: pointer;
    }
    .ppc-pin:hover svg {
        transform: scale(1.18);
        filter: drop-shadow(0 0 6px rgba(125, 243, 255, 1))
                drop-shadow(0 0 16px rgba(0, 170, 255, 0.7));
    }
</style>
<script>
window.ppcCluster = {
    groups: {},
    THEMES: {
        csv: { color: '#2dd4bf', clusterClass: '' },
        ppc: { color: '#00e5ff', clusterClass: ' ppc-own' },
    },
    // Pin PPC: picătură cyan electric cu contur deschis și centru slate cu fulger (încărcare EV)
    ppcPinIcon() {
        return L.divIcon({
            className: 'ppc-pin',
            iconSize: [30, 42],
            iconAnchor: [15, 42],
            html: `<svg viewBox="0 0 30 42" xmlns="http://www.w3.org/2000/svg">
                <path d="M15 1C7.3 1 1 7.2 1 14.9 1 25.4 15 41 15 41s14-15.6 14-26.1C29 7.2 22.7 1 15 1z"
                      fill="#00d4ff" stroke="#ccf9ff" stroke-width="1.6"/>
                <circle cx="15" cy="15" r="8.2" fill="#0f172a"/>
                <path d="M16.4 8.8 11.2 16h3.3l-1 6.2 5.3-7.4h-3.4z" fill="#7df3ff"/>
            </svg>`,
        });
    },
    // key: 'csv' | 'ppc'  — fiecare cheie are propriul L.markerClusterGroup
    show(mapId, key, stations, eventName) {
        this.hide(mapId, key);
        const map = getElement(mapId).map;
        const theme = this.THEMES[key];
        const group = L.markerClusterGroup({
            disableClusteringAtZoom: 14,
            maxClusterRadius: 60,
            spiderfyOnMaxZoom: true,
            showCoverageOnHover: true,
            polygonOptions: { color: theme.color, weight: 1.5, opacity: 0.8, fillOpacity: 0.08 },
            chunkedLoading: true,
            iconCreateFunction(cluster) {
                const n = cluster.getChildCount();
                const [size, cls] = n < 10 ? [36, 'sm'] : n < 100 ? [44, 'md'] : [54, 'lg'];
                return L.divIcon({
                    html: '<div>' + n + '</div>',
                    className: 'ppc-cluster ppc-cluster-' + cls + theme.clusterClass,
                    iconSize: L.point(size, size),
                });
            },
        });
        const markers = stations.map(([idx, lat, lon]) => {
            const m = key === 'ppc' ? L.marker([lat, lon], { icon: this.ppcPinIcon() }) : L.marker([lat, lon]);
            m.on('click', () => emitEvent(eventName, { idx, lat, lon }));
            return m;
        });
        group.addLayers(markers);
        map.addLayer(group);
        this.groups[mapId + ':' + key] = group;
    },
    hide(mapId, key) {
        const id = mapId + ':' + key;
        const group = this.groups[id];
        if (!group) return;
        const el = getElement(mapId);
        if (el && el.map) el.map.removeLayer(group);
        delete this.groups[id];
    },
};
</script>
''')

ui.query('body').style('background-color: #0f172a; color: #f8fafc; font-family: sans-serif;')

# STRATURILE AGENTULUI AI pe hartă: contur zonă, recomandări numerotate, stații afectate, punct evaluat.
# Datele vin din răspunsul /api/agent/run (câmpul `harta`, construit din rezultatele tool-urilor).
ui.add_head_html('''
<style>
    .agent-pin { background: transparent; border: none; }
    .agent-pin > div {
        width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
        border-radius: 50%; font-weight: 800; font-family: ui-monospace, monospace; font-size: 13px;
        box-shadow: 0 0 8px rgba(0,0,0,0.6);
    }
    .agent-rec > div { background: #f59e0b; color: #0f172a; border: 2px solid #fef3c7; }
    .agent-point > div { background: #a855f7; color: #fff; border: 2px solid #f3e8ff; }
    .agent-report h2 { font-size: 14px; font-weight: 700; color: #5eead4; margin: 10px 0 4px; }
    .agent-report h3 { font-size: 13px; font-weight: 700; color: #99f6e4; margin: 8px 0 2px; }
    .agent-report p, .agent-report li { font-size: 12px; line-height: 1.45; }
    .agent-report ul { list-style: disc; padding-left: 18px; }
    .agent-report table { border-collapse: collapse; font-size: 11px; margin: 4px 0; width: 100%; }
    .agent-report th, .agent-report td { border: 1px solid #334155; padding: 2px 4px; text-align: left; }
    .agent-report th { background: #1e293b; color: #cbd5e1; }
</style>
<script>
window.agentLayer = {
    layers: {},
    clear(mapId) {
        const g = this.layers[mapId];
        if (!g) return;
        const el = getElement(mapId);
        if (el && el.map) el.map.removeLayer(g);
        delete this.layers[mapId];
    },
    pin(label, cls) {
        return L.divIcon({ className: 'agent-pin ' + cls, iconSize: [30, 30], iconAnchor: [15, 15],
                           html: '<div>' + label + '</div>' });
    },
    show(mapId, h) {
        this.clear(mapId);
        const map = getElement(mapId).map;
        const g = L.featureGroup();
        if (h.contur) {
            L.geoJSON(h.contur, { style: { color: '#2dd4bf', weight: 2, dashArray: '6 4', fillOpacity: 0.04 } }).addTo(g);
        }
        (h.statii || []).forEach(([lat, lon, cod]) =>
            L.circleMarker([lat, lon], { radius: 4, color: '#94a3b8', weight: 1, fillOpacity: 0.7 })
             .bindTooltip('Stație existentă ' + cod).addTo(g));
        (h.afectate || []).forEach(a =>
            L.circleMarker([a.lat, a.lon], { radius: 8, color: '#f43f5e', weight: 2, fillOpacity: 0.25 })
             .bindTooltip(a.cod + ': ' + a.delta + ' ore/zi (canibalizare)').addTo(g));
        (h.recomandari || []).forEach(r =>
            L.marker([r.lat, r.lon], { icon: this.pin(r.eticheta, 'agent-rec'), zIndexOffset: 20000 })
             .bindTooltip('#' + r.eticheta + ' · ' + r.ore + ' ore încărcare/zi · ' + r.id).addTo(g));
        if (h.punct) {
            L.marker(h.punct, { icon: this.pin('★', 'agent-point'), zIndexOffset: 21000 })
             .bindTooltip('Punct evaluat').addTo(g);
        }
        g.addTo(map);
        this.layers[mapId] = g;
        let b = null;
        if (h.bbox) b = L.latLngBounds([[h.bbox[1], h.bbox[0]], [h.bbox[3], h.bbox[2]]]);
        else if (g.getLayers().length) b = g.getBounds();
        if (b && b.isValid()) map.fitBounds(b, { padding: [40, 40], maxZoom: 13 });
    },
};
</script>
''')

selected_station = {"data": None}


def clear_selection():
    """Deselectează stația curentă => station_details_panel nu mai randează nimic."""
    if selected_station["data"] is None:
        return
    selected_station["data"] = None
    station_details_panel.refresh()


# =====================================================================
# AGENT AI (Gemini + tool-uri deterministe) — pornit din meniul „Agent AI”
# =====================================================================
# Agentul rulează în API (POST /api/agent/run); aici doar cerem acțiunea, afișăm raportul
# și desenăm straturile pe hartă. Toate cifrele din raport vin din tool-uri (vezi „De unde vin cifrele”).
from nicegui import run as ng_run
import nicegui_api_client as agent_api

TARA_INTREAGA = 'România (Toată Țara)'
ECON_FIELDS = {
    'pret_kwh': 'Preț vânzare (RON/kWh)',
    'cost_energie_kwh': 'Cost energie (RON/kWh)',
    'capex_per_punct': 'Investiție / punct (RON)',
    'factor_putere_medie': 'Factor putere medie (0-1)',
    'factor_ajustare_cerere': 'Ajustare cerere piață RO (0-1)',
}
agent_state = {
    'status': 'idle',  # idle | loading | done | error
    'titlu': '', 'mesaj': '', 'rezultat': None, 'pick_point': False,
    'cfg': {'k': 3, 'n_piles': 2, 'power_kw': 50.0, 'min_dist_m': 1000, 'min_dist_statie_m': 0,
            **{key: None for key in ECON_FIELDS}},
}


def _agent_params(extra):
    """Configurația curentă (n_piles, power_kw, ipoteze economice completate) + parametrii acțiunii."""
    cfg = agent_state['cfg']
    params = {'n_piles': int(cfg['n_piles']), 'power_kw': float(cfg['power_kw'])}
    params.update({key: float(cfg[key]) for key in ECON_FIELDS if cfg.get(key) not in (None, '')})
    params.update(extra)
    return params


def _show_report():
    agent_report.refresh()
    report_card.set_visibility(True)


async def run_agent_action(actiune, parametri, titlu):
    if agent_state['status'] == 'loading':
        ui.notify('Agentul rulează deja o analiză.', type='warning')
        return
    agent_state.update(status='loading', titlu=titlu, mesaj='', rezultat=None)
    _show_report()
    try:
        res = await ng_run.io_bound(agent_api.run_agent, actiune, parametri)
    except agent_api.AgentError as e:
        agent_state.update(status='error', mesaj=str(e))
        _show_report()
        return
    except Exception as e:  # eroare neașteptată: o arătăm, nu oprim aplicația
        agent_state.update(status='error', mesaj=f'Eroare neașteptată: {e}')
        _show_report()
        return
    agent_state.update(status='done', rezultat=res)
    _show_report()
    await map_widget.initialized()
    ui.run_javascript(f"window.agentLayer.show({map_widget.id}, {json.dumps(res.get('harta') or {})})")


async def save_recommendations():
    res = agent_state.get('rezultat') or {}
    items = res.get('recomandari_salvabile') or []
    if not items:
        return
    try:
        out = await ng_run.io_bound(agent_api.save_candidates, items)
    except Exception as e:
        ui.notify(f'Salvarea a eșuat: {e}', type='negative')
        return
    ui.notify(f"{out['salvati']} locații salvate ca candidați (status „propus”).", type='positive')


def close_report():
    report_card.set_visibility(False)


def clear_agent():
    agent_state.update(status='idle', rezultat=None, pick_point=False)
    report_card.set_visibility(False)
    point_hint.set_visibility(False)
    ui.run_javascript(f"window.agentLayer.clear({map_widget.id})")


def start_point_mode():
    agent_state['pick_point'] = True
    point_hint.set_visibility(True)
    cfg = agent_state['cfg']
    ui.notify(f"Faceți click pe hartă în punctul de evaluat (stație {cfg['n_piles']} x {cfg['power_kw']:g} kW).",
              type='info')


def cancel_point_mode():
    agent_state['pick_point'] = False
    point_hint.set_visibility(False)


async def analyze_insight_point(lat, lon, localitate=None):
    """Click pe coordonatele unei oportunități: centrează harta și rulează evaluarea punctului."""
    map_widget.set_center((lat, lon))
    map_widget.set_zoom(14)
    titlu = f"Evaluare {localitate + ' · ' if localitate else ''}{lat:.4f}, {lon:.4f}"
    await run_agent_action('evalueaza_punct', _agent_params({'lat': round(lat, 6), 'lon': round(lon, 6)}), titlu)


async def compare_counties():
    await run_agent_action('compara_judete', _agent_params({}), 'Unde merită investit: comparație între județe')


def open_suggest_dialog():
    cfg = agent_state['cfg']
    zona_implicita = county_select.value if county_select.value and county_select.value != TARA_INTREAGA else ''
    field = 'dark outlined dense color=teal'
    with ui.dialog() as dlg, ui.card().classes('bg-slate-900 text-white w-[440px] gap-2'):
        ui.label('Sugerează stații noi').classes('text-lg font-bold text-teal-300')
        ui.label('Agentul alege locațiile cu cel mai mare câștig net: cererea prezisă de model minus '
                 'cât pierd stațiile vecine (canibalizare).').classes('text-xs text-slate-400')
        zona = ui.input('Județ sau localitate', value=zona_implicita,
                        placeholder='ex. Cluj, Cluj-Napoca, județul Iași').props(field).classes('w-full')
        with ui.row().classes('w-full no-wrap gap-2'):
            k = ui.number('Câte stații', value=cfg['k'], min=1, max=10, step=1, format='%d').props(field).classes('flex-1')
            dist = ui.number('Distanță între ele (m)', value=cfg['min_dist_m'], min=0, step=250,
                             format='%d').props(field).classes('flex-1')
        with ui.row().classes('w-full no-wrap gap-2'):
            n = ui.number('Puncte / stație', value=cfg['n_piles'], min=1, max=20, step=1,
                          format='%d').props(field).classes('flex-1')
            pw = ui.select({22.0: 'AC 22 kW', 50.0: 'DC 50 kW', 150.0: 'DC 150 kW'}, value=cfg['power_kw'],
                           label='Putere / punct').props(field).classes('flex-1')
        dist_st = ui.number('Distanță minimă față de stațiile existente (m)', value=cfg['min_dist_statie_m'],
                            min=0, step=250, format='%d').props(field).classes('w-full')
        with ui.expansion('Ipoteze economice (opțional)', icon='payments').classes('w-full text-sm text-slate-300'):
            ui.label('Câmpurile goale folosesc valorile implicite, ilustrative.').classes('text-[11px] text-slate-400')
            econ = {key: ui.number(label, value=cfg.get(key)).props(field).classes('w-full')
                    for key, label in ECON_FIELDS.items()}
        ui.label('Numărul de puncte nu e optimizat de model (efectul lui nu e cauzal): alegeți-l dvs.') \
            .classes('text-[11px] text-amber-300/80')

        async def go():
            zona_txt = (zona.value or '').strip()
            if not zona_txt:
                ui.notify('Alegeți un județ sau o localitate.', type='warning')
                return
            cfg.update(k=int(k.value or 3), min_dist_m=float(dist.value or 0), n_piles=int(n.value or 2),
                       power_kw=float(pw.value or 50), min_dist_statie_m=float(dist_st.value or 0),
                       **{key: inp.value for key, inp in econ.items()})
            dlg.close()
            await run_agent_action('sugereaza_statii', _agent_params({
                'zona': zona_txt, 'k': cfg['k'], 'min_dist_m': cfg['min_dist_m'],
                'min_dist_statie_m': cfg['min_dist_statie_m']}), f'Stații noi: {zona_txt}')

        with ui.row().classes('w-full justify-end gap-2 mt-2'):
            ui.button('Anulează', on_click=dlg.close).props('flat color=white')
            ui.button('Rulează agentul', icon='auto_awesome', on_click=go).props('color=teal')
    dlg.open()


@ui.refreshable
def agent_report():
    st = agent_state
    with ui.row().classes('w-full items-center justify-between no-wrap'):
        with ui.row().classes('items-center gap-2 no-wrap'):
            ui.icon('auto_awesome', size='18px').classes('text-teal-300')
            ui.label(st['titlu'] or 'Agent AI').classes('text-sm font-bold text-teal-200')
        ui.button(icon='close', on_click=close_report).props('dense flat round color=white size=sm')
    if st['status'] == 'loading':
        with ui.row().classes('items-center gap-3 py-4'):
            ui.spinner('dots', size='lg', color='teal')
            ui.label('Agentul analizează datele… (de obicei 10–60 s)').classes('text-xs text-slate-300')
    elif st['status'] == 'error':
        with ui.row().classes('items-start gap-2 no-wrap'):
            ui.icon('error_outline', size='20px').classes('text-rose-400')
            ui.label(st['mesaj']).classes('text-sm text-rose-200 whitespace-pre-wrap')
    elif st['status'] == 'done' and st['rezultat']:
        res = st['rezultat']
        ui.markdown(res['raport'], extras=['tables']).classes('agent-report w-full text-slate-200')
        if res.get('recomandari_salvabile'):
            ui.button('Salvează recomandările ca candidați', icon='bookmark_add',
                      on_click=save_recommendations).props('color=teal dense').classes('w-full')
        with ui.expansion(f"De unde vin cifrele ({len(res['apeluri'])} apeluri de tool-uri)",
                          icon='fact_check').classes('w-full text-xs text-slate-300'):
            for c in res['apeluri']:
                args = json.dumps(c['argumente'], ensure_ascii=False)
                ui.label(f"{'✓' if c['ok'] else '✗'} {c['tool']}({args}) · {c.get('durata_s', '')} s") \
                    .classes('text-[11px] font-mono text-teal-200 break-all')
                ui.label(c['rezumat']).classes('text-[10px] font-mono text-slate-400 break-all mb-1')
        tok = res.get('tokeni') or {}
        tok_txt = f" · tokeni: {tok.get('intrare')} intrare / {tok.get('iesire')} ieșire" if tok.get('total') else ''
        ui.label(f"Model: {res['model']} · {res['durata_s']} s{tok_txt} · cifrele provin din tool-uri, nu din LLM") \
            .classes('text-[10px] text-slate-500')


# HEADER
with ui.header().classes('bg-slate-900 border-b border-slate-800 text-white px-6 py-3 flex justify-between items-center'):
    with ui.row().classes('items-center gap-3'):
        ui.icon('ev_station', size='32px').classes('text-teal-400')
        with ui.column().classes('gap-0'):
            ui.label('PPC EV AI Agent').classes('text-xl font-bold tracking-wide text-white')
            ui.label('Network Expansion, Upscaling & Location Optimizer').classes('text-xs text-slate-400')
    with ui.row().classes('items-center gap-4'):
        with ui.button('Agent AI', icon='auto_awesome').props('color=teal unelevated no-caps'):
            with ui.menu().props('dark').classes('bg-slate-800 text-white'):
                ui.menu_item('Sugerează stații noi în județ / localitate…', on_click=open_suggest_dialog)
                ui.menu_item('Evaluează un punct pe hartă (click)', on_click=start_point_mode)
                ui.menu_item('Compară județele: unde merită investit?', on_click=compare_counties)
                ui.separator()
                ui.menu_item('Șterge rezultatele agentului de pe hartă', on_click=clear_agent)
        ui.badge(f'Stații Înregistrate Național: {len(ALL_RO_STATIONS)}', color='slate-700').classes('px-3 py-1 text-xs')

    

# LAYOUT
with ui.row().classes('w-full h-[calc(100vh-70px)] p-4 gap-4 no-wrap items-stretch overflow-hidden'):
    
    # --- SIDEBAR STÂNGA ---
    with ui.column().classes('w-1/3 min-w-[260px] max-w-[400px] h-full gap-4 overflow-y-auto pr-1 flex-shrink-0'):
        
        with ui.card().classes('w-full bg-slate-900 border border-slate-800 p-4 rounded-xl'):
            ui.label('PPC Executive Summary').classes('text-sm font-semibold text-slate-400 mb-2')
            
            with ui.grid(columns=2).classes('w-full gap-3 text-center'):
                with ui.column().classes('bg-slate-800/60 p-2 rounded-lg border border-slate-700/50'):
                    ui.label('Stații PPC').classes('text-xs text-slate-400')
                    ui.label(str(len(PPC_STATIONS))).classes('text-xl font-bold text-teal-400')
                with ui.column().classes('bg-slate-800/60 p-2 rounded-lg border border-slate-700/50'):
                    ui.label('Total Stații RO').classes('text-xs text-slate-400')
                    ui.label(str(len(ALL_RO_STATIONS))).classes('text-xl font-bold text-amber-400')
                with ui.column().classes('bg-slate-800/60 p-2 rounded-lg border border-slate-700/50'):
                    ui.label('Putere Medie / Stație').classes('text-xs text-slate-400')
                    ui.label(f"{STATS['avg_power']} kW").classes('text-xs font-bold text-sky-400')

                with ui.column().classes('bg-slate-800/60 p-2 rounded-lg border border-slate-700/50'):
                    ui.label('Stații Open 24/7').classes('text-xs text-slate-400')
                    ui.label(str(STATS['open_24'])).classes('text-xs font-bold text-emerald-400')

        # CONTROALE: SELECTOR JUDEȚE + TOGGLE TOATE STAȚIILE
        with ui.card().classes('w-full bg-slate-900 border border-slate-800 p-4 rounded-xl gap-3'):
            ui.label('Filtre & Vizualizare Hartă').classes('text-sm font-semibold text-slate-400')
            
            def on_county_change(e):
                if e.value in ROMANIAN_COUNTIES:
                    lat, lng, zoom = ROMANIAN_COUNTIES[e.value]
                    map_widget.set_center((lat, lng))
                    map_widget.set_zoom(zoom)

            county_select = ui.select(
                options=list(ROMANIAN_COUNTIES.keys()),
                value='România (Toată Țara)',
                with_input=True,
                on_change=on_county_change
            ).props('outlined dense dark color=teal text-color=white').classes('w-full bg-slate-800 rounded')

            ui.separator().classes('bg-slate-800 my-1')

            # SWITCH PENTRU AFIȘARE STAȚII PPC (dezactivat la pornire)
            # Stațiile PPC au propriul L.markerClusterGroup (cheia 'ppc' în window.ppcCluster),
            # independent de grupul 'csv' al tuturor stațiilor din România.
            async def toggle_ppc_stations(e):
                await map_widget.initialized()
                if e.value:
                    payload = [[i, st['lat'], st['lng']] for i, st in enumerate(PPC_STATIONS)]
                    ui.run_javascript(
                        f"window.ppcCluster.show({map_widget.id}, 'ppc', {json.dumps(payload)}, 'ppc_marker_click')"
                    )
                else:
                    ui.run_javascript(f"window.ppcCluster.hide({map_widget.id}, 'ppc')")
                    clear_selection()

            ui.switch(
                'Afișează stațiile PPC',
                value=False,
                on_change=toggle_ppc_stations
            ).props('color=cyan').classes('text-xs text-slate-300 font-medium')

            # SWITCH PENTRU AFIȘARE TOATE STAȚIILE RO
            # Pinii sunt grupați într-un L.markerClusterGroup creat din JavaScript (vezi window.ppcCluster).
            # Trimitem doar [index, lat, lon]; la click, JS emite 'csv_marker_click' cu indexul,
            # iar Python citește detaliile complete din ALL_RO_STATIONS.
            async def toggle_all_stations(e):
                await map_widget.initialized()
                if e.value:
                    payload = [
                        [i, float(st['lat']), float(st['lon'])]
                        for i, st in enumerate(ALL_RO_STATIONS)
                        if pd.notna(st.get('lat')) and pd.notna(st.get('lon'))
                    ]
                    ui.run_javascript(
                        f"window.ppcCluster.show({map_widget.id}, 'csv', {json.dumps(payload)}, 'csv_marker_click')"
                    )
                else:
                    ui.run_javascript(f"window.ppcCluster.hide({map_widget.id}, 'csv')")
                    clear_selection()

            ui.switch(
                'Afișează TOATE stațiile din România', 
                on_change=toggle_all_stations
            ).props('color=amber').classes('text-xs text-slate-300 font-medium')


            # LOGICĂ HEATMAP
            # Stratul e creat prin API-ul NiceGUI: generic_layer(name='heatLayer') => L.heatLayer(date, opțiuni)
            def show_heatmap():
                hide_heatmap()
                heat_data = to_heat_data(get_demand_points())
                if not heat_data:
                    return
                heat_state['layer'] = map_widget.generic_layer(
                    name='heatLayer', args=[heat_data, HEATMAP_OPTIONS]
                )

            def hide_heatmap():
                if heat_state['layer'] is not None:
                    map_widget.remove_layer(heat_state['layer'])
                    heat_state['layer'] = None
                heatmap_info.set_text('')

            def toggle_heatmap(e):
                show_heatmap() if e.value else hide_heatmap()

            # SWITCH VIZUAL ÎN SIDEBAR
            heatmap_switch = ui.switch(
                'Afișează Heatmap Cerere Optimizată',
                on_change=toggle_heatmap
            ).props('color=rose').classes('text-xs text-slate-300 font-medium mt-2')

            # Legendă heatmap (vizibilă doar când switch-ul e pornit)
            with ui.column().classes('w-full gap-1').bind_visibility_from(heatmap_switch, 'value'):
                ui.element('div').classes('w-full h-2 rounded').style(
                    'background: linear-gradient(to right, blue, cyan, lime, yellow, red)'
                )
                with ui.row().classes('w-full justify-between text-[10px] text-slate-400'):
                    ui.label('Cerere scăzută')
                    ui.label('Cerere ridicată')
                heatmap_info = ui.label('').classes('text-[10px] text-slate-500 italic')

        # PANOU DETAILS AI
        with ui.card().classes('w-full bg-slate-900 border border-slate-800 p-4 rounded-xl gap-3 flex-grow'):
            with ui.row().classes('items-center justify-between w-full'):
                # PANOU DETAILS AI - MINI-CLASAMENT DINAMIC
                with ui.card().classes('w-full bg-slate-900 border border-slate-800 p-4 rounded-xl gap-3 flex-grow'):
                    ui.label('Oportunități de Extindere').classes('text-xs font-bold text-slate-400 uppercase tracking-wider mb-1')

                    insights_list = get_top_ai_insights()
                    for idx, insight in enumerate(insights_list):
                        with ui.column().classes('w-full gap-1 py-1'):
                            with ui.row().classes('items-center justify-between w-full'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.label(insight['priority']).classes('text-xs font-mono font-bold text-teal-400')
                                    # localitatea + coordonatele locației; click pe coordonate = analiza agentului
                                    # pentru o stație nouă în acel punct (exemplele mock nu au coordonate: titlul)
                                    if insight.get('lat') is not None and insight.get('lon') is not None:
                                        lat_i, lon_i = float(insight['lat']), float(insight['lon'])
                                        with ui.row().classes('items-center gap-1 no-wrap'):
                                            ui.label(f"{insight.get('localitate') or '-'} ·") \
                                                .classes('text-xs font-bold text-slate-100')
                                            ui.label(f"{lat_i:.4f}, {lon_i:.4f}") \
                                                .classes('text-xs font-bold text-teal-300 underline cursor-pointer '
                                                         'hover:text-teal-100') \
                                                .tooltip('Analizează o stație nouă în acest punct (Agent AI)') \
                                                .on('click', lambda _, la=lat_i, lo=lon_i, loc=insight.get('localitate'):
                                                    analyze_insight_point(la, lo, loc))
                                    else:
                                        ui.label(insight['title']).classes('text-xs font-bold text-slate-100')
                            

                        # Separator fin între elemente (fără să apară sub ultimul)
                        if idx < len(insights_list) - 1:
                            ui.separator().classes('bg-slate-800/60 my-1')
                ui.icon('auto_awesome').classes('text-teal-400')
            
            # CARD DETALII PLUTITOR (DREAPTA JOS PE HARTĂ)
        @ui.refreshable
        def station_details_panel():
          st = selected_station['data']
          if not st:
                return        

          badge_color = (
              'emerald-600'
              if st.get('ai_action') == 'UPSCALE'
              else (
                  'rose-600' if st.get('ai_action') == 'RELOCATE' else 'teal-600'
              )
          )

          with ui.card().classes(
              'absolute bottom-6 right-6 z-[1000] w-80 bg-slate-900/95 border'
              ' border-slate-700 p-4 rounded-xl shadow-2xl backdrop-blur gap-2'
          ):
            with ui.row().classes('justify-between items-center w-full'):
              ui.label(st.get('id', 'N/A')).classes(
                  'text-xs font-mono text-slate-400'
              )
              ui.badge(st.get('ai_action', 'PPC'), color=badge_color).classes(
                  'text-[10px] px-2'
              )

            ui.label(st.get('name', 'Stație PPC')).classes(
                'text-sm font-bold text-white leading-tight'
            )
            ui.label(
                f"Județ: {st.get('county', '-')} | Localitate: {st.get('city', '-')}"
            ).classes('text-xs text-slate-300 font-medium')

            ui.label(
                f"Configurație Porturi: {st.get('ports', 'N/A')}"
            ).classes('text-xs text-slate-400')
            ui.label(
                f"Grad Utilizare: {st.get('utilization', 'N/A')}"
            ).classes('text-xs text-teal-300 font-semibold')

            ui.separator().classes('bg-slate-700 my-1')

            ui.label('Detalii / Informații:').classes(
                'text-xs font-bold text-slate-300'
            )
            ui.label(
                st.get('ai_reason', 'Fără analize suplimentare.')
            ).classes('text-xs text-slate-400 italic leading-snug')

        station_details_panel()

    # --- DREAPTA: HARTĂ LEAFLET ---
    with ui.column().classes('w-2/3 min-w-0 h-full gap-3 flex-grow relative'):
        
        # Instanțiere Hartă 
        map_widget = ui.leaflet(
            center=(45.9432, 24.9668), 
            zoom=7,
            additional_resources=LEAFLET_PLUGINS,
        ).classes('w-full h-full min-h-[400px] rounded-xl border border-slate-800 shadow-2xl order-2 relative')

        # Panoul cu raportul agentului AI, suprapus peste hartă (ascuns până la prima rulare)
        report_card = ui.card().classes(
            'absolute top-[76px] right-4 z-[1000] w-[460px] max-w-[92%] max-h-[calc(100%-96px)] '
            'overflow-y-auto bg-slate-900/95 border border-teal-700/50 rounded-xl p-4 gap-2 shadow-2xl')
        with report_card:
            agent_report()
        report_card.set_visibility(False)

        # 3. Toolbar
        with ui.card().classes('w-full bg-slate-900 border border-slate-800 p-3 rounded-xl flex-wrap sm:flex-row items-center justify-between gap-2 order-1'):
            with ui.row().classes('items-center gap-3 flex-wrap'):
                ui.label('Hartă Interactivă - Stații Încărcare').classes('text-sm font-semibold text-white leading-tight')
                with ui.row().classes('gap-1'):
                    ui.button(icon='add', on_click=lambda: map_widget.set_zoom(map_widget.zoom + 1)).props('dense flat color=white')
                    ui.button(icon='remove', on_click=lambda: map_widget.set_zoom(map_widget.zoom - 1)).props('dense flat color=white')

            # Indicator pentru modul „Evaluează un punct pe hartă” (click pe el = anulare)
            point_hint = ui.button('Click pe hartă pentru evaluare · Anulează', icon='ads_click',
                                   on_click=cancel_point_mode).props('dense color=purple no-caps')
            point_hint.set_visibility(False)

            with ui.row().classes('items-center gap-2 text-xs font-mono text-slate-400 bg-slate-800 px-2.5 py-1 rounded-lg border border-slate-700 flex-wrap'):
                ui.label().bind_text_from(map_widget, 'center', backward=lambda c: f'GPS: {c[0]:.2f}, {c[1]:.2f}' if c else 'GPS: -')
                # ui.label('•').classes('text-slate-600')
                ui.label().bind_text_from(map_widget, 'zoom', backward=lambda z: f'Zoom: {z}' if z else 'Zoom: -')

        # Click pe harta goală => deselectăm stația și ascundem panoul.
        # Click-urile pe pini NU ajung aici (markerele Leaflet nu propagă click-ul către hartă);
        # ele sunt tratate de on_ppc_marker_click / on_csv_marker_click.
        # În modul „Evaluează un punct”, click-ul pornește agentul pentru acel punct.
        async def on_map_click(e):
            if agent_state['pick_point']:
                ll = (e.args or {}).get('latlng') or {}
                if 'lat' in ll and 'lng' in ll:
                    cancel_point_mode()
                    lat, lon = round(ll['lat'], 6), round(ll['lng'], 6)
                    await run_agent_action('evalueaza_punct', _agent_params({'lat': lat, 'lon': lon}),
                                           f'Evaluare punct {lat:.4f}, {lon:.4f}')
                return
            clear_selection()

        def csv_station_details(st):
            operator_name = st.get('operator', 'Independentă')
            loc = st.get('localitate', 'N/A')
            jud = st.get('judet', 'N/A')
            cap = int(st.get('capacity', 0))
            return {
                "id": f"RO-EV-{operator_name.upper()}",
                "name": f"Stație {operator_name}",
                "city": loc,
                "county": jud,
                "type": f"Capacitate: {cap if cap > 0 else 'N/A'} porturi",
                "ports": f"Putere Max: {st.get('power_max_kw', 0)} kW (Porturi: {int(st.get('power_count', 0))})",
                "ai_action": "TERȚ",
                "ai_reason": f"Localitate: {loc} | Județ: {jud} | Program 24/7: {'Da' if st.get('is_open_24_7') == 1 else 'Nu/Necunoscut'}",
                "utilization": _utilization_text(st, fallback="Date Terți"),
            }

        # Click pe un pin individual din cluster (emis din JS: emitEvent('csv_marker_click', {idx, lat, lon}))
        def on_csv_marker_click(e):
            idx = e.args.get('idx') if isinstance(e.args, dict) else None
            if idx is None or not (0 <= idx < len(ALL_RO_STATIONS)):
                return
            selected_station["data"] = csv_station_details(ALL_RO_STATIONS[idx])
            station_details_panel.refresh()

        ui.on('csv_marker_click', on_csv_marker_click)

        # Click pe un pin PPC (grup de clustering separat, culoare cyan electric)
        def on_ppc_marker_click(e):
            idx = e.args.get('idx') if isinstance(e.args, dict) else None
            if idx is None or not (0 <= idx < len(PPC_STATIONS)):
                return
            selected_station["data"] = PPC_STATIONS[idx]
            station_details_panel.refresh()

        ui.on('ppc_marker_click', on_ppc_marker_click)

        # Stațiile PPC NU mai sunt adăugate automat la 'init' — sunt controlate
        # de switch-ul „Afișează stațiile PPC” din sidebar (toggle_ppc_stations).
        map_widget.on('map-click', on_map_click)

# RULARE APLICAȚIE
ui.run(
    title='PPC EV Network AI Agent',
    dark=True,
    port=int(__import__('os').getenv('NICEGUI_PORT', '8080')),
    reload=False
)