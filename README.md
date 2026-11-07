# UrbanEV Backend v2 — FastAPI, PostgreSQL, PostGIS, H3

## Modelul de date

### `celule_h3`
Stochează poligonul și centrul fiecărei celule H3. Sunt salvate celulele de rezoluție 8 și 9.

### `statie_incarcare`
Stochează geometria sursă `locatie`, coordonatele generate automat de PostgreSQL,
indecșii `h3_res8` și `h3_res9`, `charge_count`, `taz_id`, `area`, `perimeter`.

### `date_temporale_statie`
Stochează câte o observație per stație și timestamp:
`busy`, `volume`, `e_price`, `s_price`.
Tabelul este partiționat RANGE după `timestamp`, iar API-ul creează partiții lunare.

## Formula volumului

`volume_kwh = busy * pump_rating_kw * snapshot_minutes / 60`

Exemplu pentru 50 kW/pompă și 5 minute:
- busy=1 -> 4.166667 kWh
- busy=2 -> 8.333333 kWh
- busy=3 -> 12.500000 kWh

`PUMP_RATING_KW` este configurabil în `.env`. Valoarea 50 kW este doar o valoare inițială de configurare; înlocuiește-o cu rating-ul ales pentru proiect. Dacă stațiile au rating-uri diferite, se poate adăuga ulterior `pump_rating_kw` la nivel de stație sau pompă.

Backendul interpretează fiecare snapshot ca o stare constantă pe următoarele 5 minute. Nu calculează energia dintre două citiri prin interpolare.

## Corecții de consistență aplicate DDL-ului transmis

1. Am eliminat backtick-urile din jurul declarației `charge_count`; backtick-urile nu sunt sintaxă PostgreSQL.
2. Am adăugat `taz_id`, `area` și `perimeter`, deoarece DDL-ul le folosea în indexuri și constrângeri fără să le declare.
3. Am înlocuit referințele `occupancy` și `duration` din CHECK, deoarece tabelul temporal transmis folosește `busy`, `volume`, `e_price`, `s_price`.
4. Am păstrat `busy` ca număr de pompe ocupate și `volume` ca energie estimată în kWh.
5. Cheia primară temporală este `(statie_id, timestamp)`, compatibilă cu partiționarea RANGE.

## Instalare

Activează PostGIS și execută schema:

```bash
createdb urbanev
psql -d urbanev -f sql/schema.sql
```

Instalează dependențele și configurează `.env`:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
```

Importă stațiile din `inf.csv`:

```bash
python ingest_stations.py inf.csv
```

Pornește FastAPI:

```bash
uvicorn app.main:app --reload --port 8000
```

Documentație: `http://127.0.0.1:8000/docs`

Pornește NiceGUI separat:

```bash
python main.py
```

## Snapshot la fiecare 5 minute

Pentru fiecare stație, colectorul trimite către:

`POST /api/stations/{station_id}/snapshot`

Exemplu JSON:

```json
{
  "timestamp": "2026-11-07T12:00:00Z",
  "busy": 3,
  "e_price": 1.25,
  "s_price": 0.45
}
```

API-ul:
1. verifică existența stației;
2. verifică `busy <= charge_count`;
3. folosește rating-ul din `.env` dacă nu este transmis unul în request;
4. calculează volumul pentru următoarele 5 minute;
5. creează partiția lunară dacă nu există;
6. inserează sau actualizează observația pentru cheia `(statie_id, timestamp)`.

Răspunsul include valoarea calculată și formula utilizată.

## Endpoint-uri

- `GET /health`
- `GET /api/stations`
- `GET /api/stations/{station_id}`
- `GET /api/stats`
- `GET /api/heatmap?resolution=8|9`
- `GET /api/h3/cells?resolution=8|9`
- `GET /api/h3/cells/{h3_index}`
- `GET /api/insights?resolution=8|9`
- `POST /api/stations`
- `POST /api/stations/{station_id}/snapshot`

## Integrarea în `main.py`

Înlocuiește funcțiile locale care citesc CSV-ul sau generează punctele heatmap cu:

```python
from nicegui_api_client import (
    load_all_stations,
    get_csv_stats,
    get_demand_points,
    get_top_ai_insights,
)

ALL_RO_STATIONS = load_all_stations()
STATS = get_csv_stats(ALL_RO_STATIONS)
```

Adapterul păstrează cheile `lat` și `lon` folosite de codul Leaflet din frontend.
Datele care nu există în schema transmisă (operator, județ, putere individuală,
program 24/7) nu sunt inventate.

`PPC_STATIONS` din frontend rămâne mock până la introducerea unei tabele dedicate.
`/api/insights` face agregări descriptive H3, nu predicții ML.


## Date sintetice pentru demonstrație

Pentru a popula baza de date cu date care pot fi afișate imediat în dashboard, rulează
scriptul `seed_synthetic_data.py` după ce ai creat schema și ai configurat `.env`:

```bash
python seed_synthetic_data.py
```

Configurația implicită generează:
- 36 de stații demonstrative distribuite în mai multe zone din București;
- câte un snapshot la fiecare 5 minute pentru ultimele 48 de ore;
- aproximativ 20.772 de observații temporale;
- valori sintetice pentru `busy`, `e_price` și `s_price`;
- `volume` calculat în kWh cu formula `busy * PUMP_RATING_KW * 5 / 60`;
- celule H3 de rezoluție 8 și 9 și geometrii PostGIS asociate.

Parametri opționali:

```bash
python seed_synthetic_data.py --stations 50 --hours 72 --seed 7
```

Scriptul folosește ID-uri începând de la `900001`. La o rerulare, regenerează
observațiile pentru intervalul de ID-uri respectiv, fără să șteargă stațiile cu alte ID-uri.
Toate coordonatele, stațiile, ocupările și prețurile generate sunt fictive și sunt
destinate exclusiv testării vizualizării; nu reprezintă infrastructură sau utilizare reală.

## Pornirea frontendului conectat la PostgreSQL

Pornește API-ul, apoi generează datele sintetice și abia după aceea pornește NiceGUI:

```bash
uvicorn app.main:app --reload --port 8000
python seed_synthetic_data.py
python main.py
```

`main.py` încearcă să citească stațiile, heatmap-ul și agregările H3 din FastAPI.
Dacă API-ul nu este disponibil, păstrează fallback-ul existent către CSV/date mock.
Datele PPC din secțiunea separată a interfeței rămân demonstrative și nu sunt
înlocuite de stațiile din tabelul `statie_incarcare`.
