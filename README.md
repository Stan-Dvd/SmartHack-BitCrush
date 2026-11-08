# UrbanEV Backend v3: FastAPI, PostgreSQL/PostGIS, H3 și modelul ML de cerere

Baza de date e construită în jurul modelului LightGBM din [ml/](ml/README.md). Modelul prezice câte puncte de încărcare sunt ocupate în medie la o stație, pentru fiecare oră și tip de zi. Baza îl folosește în două moduri:

1. **Utilizarea viitoare a stațiilor existente.** Fiecare stație are un profil prezis de 24 de ore × zi lucrătoare/weekend. Profilul e proiectat pe calendar (ora locală, București) și, dacă stația are observații reale, e calibrat cu ele.
2. **Performanța unei stații noi** în orice punct din România. Feature-urile modelului (populație, POI, concurență) se calculează live, în PostGIS. Răspunsul include și cât pierd stațiile existente din jur (canibalizare). În plus, o grilă H3 de oportunități acoperă toate zonele populate.

## Modelul de date

| Tabel | Ce conține |
|---|---|
| `statie_incarcare` | cele 1.181 de stații din România (`data/romania/statii_electrice_ML_cu_localitate.csv`): `cod` (RO0001…), locație, județ, operator, `n_piles`, `power_kw`, celule H3 |
| `locatie_candidat` | locații propuse și evaluate cu modelul (`status`: propus / evaluat / aprobat / respins) |
| `poi_osm` | 132.222 de POI OpenStreetMap în categoriile modelului (food / business_res / lifestyle), EPSG:3035 |
| `populatie_grid` | grila WorldPop 2020 la 1 km: `populatie` (brută) și `populatie_ajustata` (la totalul INS al județului) |
| `judet`, `judet_bucata` | contururile județelor (OSM), populația INS 2026 și raportul WorldPop/INS |
| `localitate` | 3.193 de municipii, orașe și comune (OSM `admin_level=8`), legate de județ; folosite de agent ca să găsească o zonă după nume |
| `model_ml` | registrul modelelor: fișier, ordinea feature-urilor, parametri, metrici de test, limitări; un singur model `activ` |
| `features_locatie` | feature-urile modelului pentru o stație, un candidat sau centrul unei celule H3 (aceleași nume ca în `ml/src/config.py`) |
| `predictie_profil` | 48 de rânduri per stație sau candidat: `busy_pred` (puncte ocupate), `ocupare_pred`, `kwh_pred` |
| `predictie_sumar` | ore de încărcare pe zi, kWh pe zi, ocupare medie și la vârf, ora de vârf, saturare prezisă, percentila națională |
| `date_temporale_statie` | observațiile reale la 5 minute (`busy`, `volume`, prețuri), partiționate lunar; folosite pentru calibrare și monitorizare |
| `celule_h3` | poligoanele celulelor H3 folosite |

Funcții și view-uri SQL:

- `fn_features_punct(lon, lat, exclude_statie_id, extra_locatie, extra_piles)`: cele 9 feature-uri de locație pentru orice punct. `extra_locatie` simulează o stație nouă în concurență (pentru canibalizare).
- `fn_prognoza_statie(statie_id, de_la, pana_la, calibrat)`: prognoza orară. Factorul de calibrare = busy observat / busy prezis în ultimele 28 de zile, limitat la [0,25; 4]. Se aplică doar cu cel puțin 7 zile de observații.
- `v_statie_predictie`: stațiile cu feature-urile și sumarul predicției.
- `v_calibrare_statie`: observat vs prezis, pe stație × oră × tip de zi.

### Cum se calculează predicția

```
fn_features_punct (PostGIS)  ->  9 feature-uri + n_piles + power_kw
app/ml_service.py (LightGBM) ->  ocupare prezisă × n_piles, pentru 24 de ore × {zi lucrătoare, weekend}
                             ->  predictie_profil (48 de rânduri) + predictie_sumar
```

- **Ore de încărcare pe zi** = (5·Σ zi lucrătoare + 2·Σ weekend) / 7, ca în evaluarea modelului.
- **kWh** = puncte ocupate × `power_kw` × 1 h. E un plafon superior: un punct ocupat nu trage tot timpul puterea maximă.
- Ordinea feature-urilor și modul țintei vin din `ml/models/params.json`.

## Instalare și încărcarea datelor

PostgreSQL cu PostGIS rulează în Docker, pe portul 5433:

```bash
docker compose up -d
docker exec -i urbanev-db psql -U postgres -d urbanev < sql/schema.sql   # ATENȚIE: șterge tot

python -m venv .venv
.venv\Scripts\activate                       # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-etl.txt
copy .env.example .env
```

Datele brute nu sunt în repository (OSM și WorldPop sunt mari). Ordinea ETL contează:

```bash
python -m etl.import_statii                                        # data/romania/statii_electrice_ML_cu_localitate.csv
python -m etl.import_populatie D:\hack\romania_worldpop.zip        # WorldPop 2020 1 km (golește grila)
python -m etl.import_judete D:\hack\romania-261105.osm.pbf "D:\hack\exportPivot_POP107D (1).csv"   # după populație
python -m etl.import_poi D:\hack\romania-261105.osm.pbf
python -m etl.register_model                                       # ml/models/lgbm.txt -> model_ml (activ)
python -m etl.recalculeaza --statii --h3                           # ~1-2 minute
python -m etl.verifica_features --spatial D:\hack\romania_spatial_features.csv   # opțional: paritate
```

`etl.recalculeaza --statii` trebuie rulat din nou după un model nou (`etl.register_model`) sau după importuri. Endpoint-ul `POST /api/stations` recalculează automat stația nouă și vecinii ei pe 3 km.

Pornire:

```bash
uvicorn app.main:app --reload --port 8000    # documentație: http://127.0.0.1:8000/docs
python main.py                               # NiceGUI; citește stațiile, heatmap-ul și insight-urile din API
```

## API

| Endpoint | Ce face |
|---|---|
| `GET /api/stations` | stațiile cu sumarul predicției; filtre `judet`, bbox, `h3_res8`, `order_by=ore_incarcare_zi` |
| `GET /api/stations/{id}` | detalii: feature-uri, profilul de 48 de ore și ultimele observații |
| `GET /api/stations/{id}/forecast?start&end&calibrat` | prognoza orară (implicit următoarele 7 zile) |
| `POST /api/stations` | adaugă o stație; calculează predicția și recalculează concurența vecinilor |
| `POST /api/stations/{id}/snapshot` | observație la 5 minute (`busy <= n_piles`; volumul folosește `power_kw` al stației) |
| `POST /api/candidates/evaluate` | `{latitude, longitude, n_piles?, power_kw?, cu_impact, salveaza}`: performanța prezisă, percentila și impactul asupra stațiilor pe 3 km |
| `GET /api/candidates`, `GET/PATCH /api/candidates/{id}` | candidații salvați și statusul lor |
| `GET /api/opportunities?judet&min_dist_statie_m&limit` | celulele H3 cu cea mai mare utilizare prezisă pentru o stație de referință |
| `GET /api/heatmap?sursa=predictie\|observatii&resolution` | harta de oportunități (agregabilă la rezoluții H3 mai mici) sau agregarea observațiilor |
| `GET /api/insights` | top oportunități la cel puțin 1 km de o stație existentă, cu explicație |
| `GET /api/stats`, `GET /api/models`, `POST /api/admin/recalculeaza` | statistici, registrul modelelor, recalculare |

Exemplu: o stație nouă de 2 × 50 kW lângă Universitate, în București.

```bash
curl -X POST localhost:8000/api/candidates/evaluate -H "Content-Type: application/json" \
     -d '{"latitude": 44.4355, "longitude": 26.1020}'
# ~18,8 ore de încărcare/zi, ocupare medie 0,39, percentila 1,00 în grila H3;
# cele 12 stații pe 3 km pierd împreună ~0,6 ore/zi
```

**Stația de referință** pentru candidați și grila H3: `PREDICTIE_N_PILES_DEFAULT` × `PREDICTIE_POWER_KW_DEFAULT`, implicit 2 × 50 kW. Locațiile se compară la aceeași configurație, pentru că efectul învățat al lui `n_piles` nu e cauzal.

## Agentul AI

Agentul sugerează locații noi și explică rezultatele. Regula de bază: **LLM-ul conduce, nu calculează**. Modelul de limbaj (Google Gemini) alege tool-urile și scrie raportul, dar **toate cifrele vin din tool-uri deterministe** (SQL + modelul LightGBM). Raportul afișează lista apelurilor („De unde vin cifrele”), iar harta e construită direct din rezultatele tool-urilor, nu din textul LLM-ului.

```
Meniul „Agent AI” (NiceGUI) → POST /api/agent/run → app/agent.py (buclă Gemini, max 8 pași)
                                                   → app/agent_tools.py (8 tool-uri) → PostGIS + LightGBM
```

**În interfață**, butonul **Agent AI** din header deschide un meniu:
- **Sugerează stații noi**: alegi județul (precompletat din filtru) sau o localitate, numărul de stații, configurația, distanțele minime și, opțional, ipotezele economice. Rezultatul e un raport, plus markeri numerotați pe hartă: locațiile propuse (portocaliu) și stațiile existente care pierd clienți (roșu). Butonul „Salvează recomandările ca candidați” le scrie în `locatie_candidat` cu status `propus`.
- **Evaluează un punct pe hartă**: următorul click pe hartă rulează simularea pentru acel punct.
- **Compară județele**: clasamentul județelor după potențialul neacoperit.

**Tool-urile** (fiecare poate fi testat fără LLM: `POST /api/tools/{nume}`; lista și schemele sunt la `GET /api/tools`):

| Tool | Ce face |
|---|---|
| `resolve_area` | găsește un județ sau o localitate după nume (fără diacritice, cu prefixe precum „județul”); la nume ambigue întoarce variantele |
| `get_existing_stations` | stațiile din zonă, operatorii și utilizarea prezisă |
| `score_candidates` | celulele H3 din zonă, ordonate după orele de încărcare pe zi prezise |
| `explain_cell` | feature-urile și top 3 contribuții SHAP (LightGBM `pred_contrib`), ca efect multiplicativ asupra ocupării, plus comparația cu media județului |
| `plan_network` | setul de k locații ales greedy, cu recalcularea concurenței după fiecare alegere; câștig net = ore proprii − pierderea vecinilor |
| `simulate_station` | cererea unei stații noi într-un punct și canibalizarea pe 3 km |
| `economics` | venit, costuri, ani până la amortizare, cu ipoteze explicite (valori implicite ilustrative în `app/config.py`) |
| `rank_areas` | clasamentul județelor: celule unde modelul prezice peste nivelul de bază și care nu sunt deservite de stații |

Exemplu fără LLM:

```bash
curl -X POST localhost:8000/api/tools/plan_network -H "Content-Type: application/json" \
     -d '{"zona": "Cluj-Napoca", "k": 3, "min_dist_m": 1500}'
```

**Configurare.** Ai nevoie de o cheie gratuită de pe https://aistudio.google.com, pusă în `.env` ca `GEMINI_API_KEY=...`. Se acceptă și numele `GOOGLE_API_KEY` sau `GOOGLE_MAPS_API_KEY`. Modelul implicit e `GEMINI_MODEL=gemini-3.6-flash` (stabil și pe nivelul gratuit). Erorile trecătoare (429/500/503/504) se reîncearcă automat de 4 ori, cu pauze din ce în ce mai mari. Dacă modelul rămâne supraîncărcat, analiza se reia cu `GEMINI_MODEL_REZERVA` (implicit `gemini-3.5-flash-lite`). După ce pui cheia, repornește API-ul. Fără cheie, la limită depășită sau la o eroare de rețea, butoanele afișează un mesaj de eroare (`/api/agent/run` întoarce 503). `GET /api/agent/status` arată dacă agentul e configurat. Pe nivelul gratuit, Google poate folosi datele trimise pentru a-și îmbunătăți produsele; aici sunt doar date publice agregate.

**Ce lasă agentul intenționat pe dinafară:** optimizarea lui `n_piles` (efectul învățat nu e cauzal) și predicția de saturare (modelul nu o prinde).

## Calitatea datelor

- **Paritate cu setul românesc.** Feature-urile calculate în PostGIS coincid cu cele generate offline: POI-urile și numărătorile de concurență sunt identice, iar diferențele pentru populație (< 0,01) și distanțe (< 0,06 m) țin de rotunjire. Predicțiile din `app/ml_service.py` coincid exact cu `ml/src/train.predict_y`.
- **WorldPop e defect pentru România.** Fișierul `ppp_ROU_2020_1km_Aggregated_UNadj` pune 8,6 milioane de oameni în județul Călărași (INS: 290.000), iar celelalte județe au doar ~40–60% din populația oficială (București: 1,05M față de 2,1M). Din acest motiv:
  - `pop_1km` folosește populația **ajustată pe județ** la totalul INS 2026 (suma pe țară: 21,68M față de 21,65M INS);
  - în județele cu raport WorldPop/INS > 2 (doar Călărași), `pop_suspect = true`, pentru că distribuția din interiorul județului rămâne greșită. Celulele marcate sunt excluse implicit din oportunități și din heatmap.
  - Setul `romania_spatial_features.csv` folosit înainte conținea valorile brute, deci era afectat de același defect.
- 273 de stații nu au operator în sursă. Prin decizie de proiect, primesc `operator = 'PPC'` (`OPERATOR_IMPLICIT` din `etl/import_statii.py`) și `operator_implicit = true`. Aplicația afișează astfel 286 de stații PPC: 13 din sursă și 273 atribuite.
- 785 de stații nu au capacitate în sursă și primesc `n_piles = 1`; 768 nu au putere și primesc 22 kW (aceleași reguli ca la generarea setului românesc). O singură stație are putere 0 și e tratată ca lipsă.

## Limite

- Modelul e antrenat pe **Shenzhen** (UrbanEV, 2022–2023) și aplicat în România fără reantrenare. Nivelurile absolute sunt orientative; clasarea locațiilor e mai robustă.
- **Modelul diferențiază doar zonele urbane dense.** În ~92% din celulele populate (24.413 din 26.438), predicția pentru stația de referință e exact nivelul de bază de 13,9 h/zi: sub populația celor mai puțin dense zone din Shenzhen, modelul dă o valoare constantă. Tool-urile agentului marchează explicit fiecare locație cu `peste_nivelul_de_baza`. Între locațiile aflate la nivelul de bază, ordinea vine din criteriul secundar (populația pe 1 km, apoi POI), nu din model. Pentru județe fără orașe mari, recomandările se bazează în mare parte pe acest criteriu.
- Valorile absolute ale cererii sunt la nivelul Shenzhen, o piață EV mult mai matură, deci calculele economice sunt optimiste. `factor_ajustare_cerere` (sau `ECON_FACTOR_CERERE` în `.env`) scalează cererea pentru piața din România.
- Ținta e **cererea servită** (puncte ocupate), nu cererea totală. Modelul nu folosește prețul, vremea sau sezonul.
- Profilul unei stații existente e static (oră × tip de zi). Pentru prognoză, singura adaptare la realitate e calibrarea cu observații, iar în prezent nu există observații reale.
- `busy` din `romania_ev_5min*.csv` e sintetic și **nu** se încarcă în `date_temporale_statie`. Modelul românesc din `ml/src/romania.py` e un control negativ și nu e folosit.
- `n_piles` și `power_kw` au un efect învățat necauzal (selection bias, vezi `ml/README.md`).
