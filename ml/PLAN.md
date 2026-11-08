# Plan proiect: model ML pentru cererea la stațiile de încărcare EV

Smarthack XI 2026, tema PPC „24h to Power PPC”.

Acest document descrie **faza 1: antrenarea și evaluarea modelului ML**. Agentul și aplicarea pe România vin ulterior și sunt în afara scopului deocamdată.

---

## 1. Obiectiv

Antrenăm un model care prezice **cererea de încărcare** la o stație, pe baza caracteristicilor locației și ale stației, pentru fiecare oră a zilei și fiecare tip de zi.

Modelul va fi folosit ulterior de un agent care recomandă unde să se construiască stații noi. Din acest motiv, **modelul nu are voie să folosească informații care nu există pentru o locație nouă** (vezi secțiunea 8, Reguli anti-leakage).

## 2. Scop

**În scop:**
- încărcarea și curățarea datelor UrbanEV (Shenzhen)
- construirea țintei din câmpul `busy`
- construirea feature-urilor spațiale, de stație și temporale
- split pe zone, baseline-uri, antrenare LightGBM
- evaluare pe setul de test, analiza erorilor, verificări de sens

**În afara scopului (deocamdată):**
- aplicarea pe România
- agentul și stratul de decizie (profit, saturare, recomandări)
- OpenStreetMap: drumuri, distanțe până la drumuri, POI din OSM
- filtrarea perioadelor de sărbătoare sau COVID
- prețul, vremea, traficul, luna sau sezonul

## 3. Cerințe din regulamentul hackathonului (obligatorii)

- Model antrenat de echipă, evaluat pe un **set de test separat** și comparat cu un **baseline**.
- Codul de antrenare e în repository, iar modelul rulează local, nu printr-un API extern.
- **Analiza erorilor:** unde greșește modelul și pe ce tipuri de cazuri.
- **Reproductibilitate:** oricine poate rula proiectul după README și obține aceleași rezultate cu scriptul de evaluare.
- Doar date publice, cu sursele menționate. Repository public, cu commit-uri pe parcurs.

## 4. Date

### 4.1 UrbanEV (sursa principală)

- Repository: https://github.com/IntelligentSystemsLab/UrbanEV
- Date brute la nivel de stație: Dryad, https://doi.org/10.5061/dryad.np5hqc04z
- Licență: CC0 1.0
- Citare: Li, H., Qu, H., Tan, X. et al. *UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction.* Scientific Data 12, 523 (2025). https://doi.org/10.1038/s41597-025-04874-4
- Perioada: 1 septembrie 2022 – 28 februarie 2023. Intervale de 5 minute în datele brute.
- 1.682 de stații și 24.798 de puncte de încărcare în datele brute (1.362 de stații după curățarea autorilor).

Fișierele pe care le folosim:

| Fișier | Folosire |
|---|---|
| `charge_5min` (date brute pe stații, câmpul `busy`) | sursa țintei |
| informațiile despre stații (coordonate, putere nominală, număr de puncte, zonă) | feature-uri de stație, calcul pe raze, split |
| `station_distance.csv` (matricea de distanțe între stații, în metri) | feature-uri de competiție |
| `poi.csv` (POI în 3 categorii: alimentație, business și rezidențial, servicii lifestyle; WGS84) | feature-uri POI |

### 4.2 WorldPop (populație)

- Grilă de 100 m, China, anul 2020 (ultimul disponibil în setul Global).
- https://www.worldpop.org. Licență CC BY 4.0, cu atribuire obligatorie.
- Se decupează doar zona Shenzhen (bounding box), pentru că fișierul pentru toată China e mare.

### 4.3 PASUL 0: inspectarea datelor înainte de orice cod de pipeline

Formatul exact al fișierelor nu e confirmat. Înainte de orice altceva, scrie un script `src/inspect_data.py` care afișează, pentru fiecare fișier: coloanele, tipurile, primele rânduri, numărul de rânduri și procentul de valori lipsă. Apoi răspunde la întrebările de mai jos și notează răspunsurile în `docs/data_notes.md`:

1. `busy` e **0/1 per punct de încărcare** sau **numărul de puncte ocupate per stație**? Formatul e „lat” (o coloană per stație) sau „lung” (coloane station, time, busy)?
2. Unde se găsesc puterea nominală și numărul de puncte per stație? Puterea e per punct (atunci trebuie agregată) sau per stație?
3. Există un ID de zonă (traffic zone) pentru fiecare stație?
4. `poi.csv` conține **puncte cu coordonate** sau **numărători agregate per zonă**?
5. Cum se potrivesc ID-urile stațiilor între fișiere?

Restul planului se adaptează după aceste răspunsuri.

## 5. Ținta (y)

**Cererea = numărul mediu de puncte ocupate**, per stație × oră × tip de zi.

```
busy la 5 min
  → număr de puncte ocupate per stație per interval
  → medie per (stație, oră 0–23, is_weekend)
  → y
```

- Unitatea: puncte ocupate în medie în acea oră, adică ore de încărcare pe oră.
- Mărimi derivate, pentru raportare: rata de ocupare `y / n_piles` și orele de încărcare pe zi, adică suma lui y pe cele 24 de ore.
- Limită de documentat: `busy` măsoară **cererea servită**. La stațiile pline, cererea reală e mai mare decât ce se vede în date.

**Tratarea datelor lipsă:**
- Media se calculează doar pe intervalele disponibile.
- Se elimină stațiile cu acoperire sub **70%** din intervalele perioadei (pragul folosit și de autorii UrbanEV; vezi `docs/data_notes.md`).
- Se elimină celulele (stație, oră, tip de zi) cu prea puține observații, de exemplu sub 10.
- Se raportează câte stații și celule au fost eliminate și de ce (în `reports/data_quality.json`).

## 6. Feature-uri

Toate distanțele sunt în metri. Pentru calculele pe raze se folosește distanța haversine din coordonate sau `station_distance.csv`. Competiția se calculează din **toate cele 1.682 de stații brute**, inclusiv cele eliminate din țintă: sunt infrastructură reală, chiar dacă nu le folosim ca exemple de antrenare.

### 6.1 Spațiale (locație)

| Feature | Definiție |
|---|---|
| `dist_nearest_station_m` | distanța până la cea mai apropiată **altă** stație (se exclude diagonala) |
| `n_stations_500m` | numărul altor stații pe o rază de 500 m |
| `n_stations_1km` | numărul altor stații pe o rază de 1 km |
| `n_stations_3km` | numărul altor stații pe o rază de 3 km |
| `competitor_piles_1km` | suma punctelor de încărcare ale altor stații pe o rază de 1 km |
| `pop_1km` | suma populației WorldPop pe o rază de 1 km |
| `poi_food_500m` | POI alimentație pe o rază de 500 m |
| `poi_business_res_500m` | POI business și rezidențial pe o rază de 500 m |
| `poi_lifestyle_500m` | POI servicii lifestyle pe o rază de 500 m |

Dacă `poi.csv` e agregat per zonă (pasul 0, întrebarea 4), feature-urile POI devin **densități per zonă** (numărul de POI / suprafața zonei), atribuite fiecărei stații după zona ei. Documentați alegerea.

### 6.2 Despre stație

| Feature | Definiție |
|---|---|
| `n_piles` | numărul de puncte de încărcare ale stației |
| `power_kw` | puterea nominală a stației (media pe puncte, dacă puterea e dată per punct) |

Mai târziu, agentul va **alege** aceste valori pentru locațiile noi. De aceea trebuie verificat că efectul lor învățat are sens (secțiunea 10).

### 6.3 Temporale

| Feature | Definiție |
|---|---|
| `hour` | ora din zi, 0–23 |
| `is_weekend` | 1 pentru sâmbătă și duminică, 0 altfel |

### 6.4 Excluse intenționat (nu se adaugă fără discuție)

| Exclus | Motiv |
|---|---|
| latitudine, longitudine, ID de zonă | nu se transferă în alte locații; modelul ar memora locuri, nu reguli |
| orice derivat din `busy` (ocupare anterioară, lag-uri, saturarea vecinilor) | nu există pentru locații noi; ar fi leakage |
| prețul | o stație nouă nu are preț; efect greu de separat de cauzalitatea inversă |
| vremea, luna, sezonul | doar 6 luni de date; nu se transferă |
| OpenStreetMap (drumuri, POI) | scos din scop |

## 7. Tabelul de antrenare

Un rând per **(stație, oră, is_weekend)**, adică aproximativ 1.400 × 24 × 2 ≈ 67.000 de rânduri.

```
station_id | zone_id | [feature-uri spațiale] | n_piles | power_kw | hour | is_weekend | y
```

`station_id` și `zone_id` sunt păstrate **doar** pentru split și analiză. Nu intră în model.

Fișierul rezultat: `data/processed/train_table.parquet`.

## 8. Reguli anti-leakage (obligatorii)

1. **Split pe zone:** toate rândurile unei stații stau ori în train, ori în test, iar toate stațiile unei zone stau împreună.
2. Fără amestecare aleatoare la nivel de rând.
3. Fără latitudine, longitudine sau ID-uri în feature-uri.
4. Fără feature-uri derivate din `busy`.
5. Orice statistică folosită în baseline-uri sau normalizări se calculează **doar pe train**.

## 9. Split, baseline-uri, model

### 9.1 Split

- **Test:** 20% din zone, ales o singură dată cu `seed = 42` și salvat în `data/processed/split.json`. Nu se atinge până la evaluarea finală.
- **Tuning:** `GroupKFold` cu 5 fold-uri pe zone, doar în train.
- Dacă nu există ID de zonă (pasul 0, întrebarea 3): grupuri create cu KMeans pe coordonate, aproximativ 50 de clustere. Coordonatele servesc doar la grupare, nu ca feature.

### 9.2 Baseline-uri

| Baseline | Predicție |
|---|---|
| **B0** | media lui y din train pentru fiecare (oră, is_weekend), adică profilul mediu al orașului |
| **B1** | rata medie de ocupare din train pentru fiecare (oră, is_weekend), înmulțită cu `n_piles` al stației |

B1 e baseline-ul principal. Dacă LightGBM îl bate clar, înseamnă că **locația** contează dincolo de mărimea stației.

### 9.3 Model

- `lightgbm.LGBMRegressor`
- Funcții obiectiv de comparat: `regression` (L2) și `tweedie`, pentru că ținta e nenegativă și asimetrică.
- Early stopping pe fold-urile de validare. Parametri de pornire: `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `feature_fraction=0.8`, `random_state=42`.
- Tuning simplu, câteva combinații. Nu e o prioritate.
- Modelul final se salvează în `models/lgbm.txt`, iar parametrii în `models/params.json`.

## 10. Evaluare

### 10.1 Metrici pe setul de test (pentru B0, B1 și LightGBM)

| Metrică | Nivel |
|---|---|
| MAE, RMSE, R² pe y | rând (stație × oră × tip zi) |
| MAE pe orele de încărcare pe zi | stație |
| corelația Spearman între cererea zilnică prezisă și cea reală | stație: contează pentru clasarea locațiilor |

Rezultatele se salvează în `reports/metrics.json`, plus un tabel comparativ în README.

### 10.2 Analiza erorilor

- Reziduuri pe grupuri: cvartile de `n_piles`, `power_kw`, `pop_1km`, `n_stations_1km`, apoi pe `hour` și `is_weekend`.
- Top 20 de stații cu cea mai mare eroare: ce au în comun?
- Comparație pe oră între profilul prezis și cel real, pentru câteva stații tipice și câteva atipice.
- Ce s-a încercat pentru corectare și cu ce rezultat (cerință din regulament). Notați în `docs/experiments.md`.

### 10.3 Explicabilitate

- SHAP summary plot pe setul de test.
- Importanța feature-urilor de competiție (`dist_nearest_station_m`, `n_stations_*`, `competitor_piles_1km`) se raportează **și grupată**, ca un singur factor „competiție”, pentru că sunt corelate și SHAP le împarte efectul.

### 10.4 Verificări de sens

- **ICE / partial dependence** pentru `n_piles` și `power_kw` (`sklearn.inspection.PartialDependenceDisplay`). Cererea prezisă trebuie să crească, tot mai încet. Dacă efectul e absurd (descrescător sau liniar fără limită), reantrenați cu `monotone_constraints` și documentați.
- **Saturare:** pentru fiecare stație din test, saturarea reală = procentul de intervale de 5 minute în care toate punctele erau ocupate. Comparați cu saturarea prezisă: rata de ocupare la ora de vârf peste 0,9. Raportați precizia și recall-ul. Saturarea prezisă e o aproximare, pentru că modelul prezice medii orare.

## 11. Structura repository-ului

```
.
├── README.md
├── PLAN.md
├── requirements.txt
├── data/
│   ├── raw/            # gitignored; populat de src/download.py
│   └── processed/
├── docs/
│   ├── data_notes.md   # răspunsurile de la pasul 0
│   └── experiments.md  # jurnalul încercărilor
├── models/
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── data_quality.json
└── src/
    ├── config.py              # căi, raze, praguri, seed
    ├── download.py            # descarcă UrbanEV și decupajul WorldPop
    ├── inspect_data.py        # pasul 0
    ├── load_urbanev.py
    ├── target.py              # busy → y
    ├── features_spatial.py
    ├── features_station.py
    ├── features_temporal.py
    ├── build_dataset.py       # → train_table.parquet
    ├── split.py               # → split.json
    ├── baselines.py
    ├── train.py
    ├── evaluate.py            # → metrics.json + tabelul comparativ
    └── analysis.py            # erori, SHAP, ICE, saturare → figures/
```

Toate constantele (raze, prag de acoperire 70%, prag de saturare 0,9, seed) stau în `src/config.py`.

## 12. Comenzi de rulare (de scris în README)

```bash
pip install -r requirements.txt
python -m src.download
python -m src.inspect_data
python -m src.build_dataset
python -m src.split
python -m src.train
python -m src.evaluate
python -m src.analysis
```

Biblioteci principale: `pandas`, `numpy`, `pyarrow`, `lightgbm`, `scikit-learn`, `shap`, `rasterio`, `matplotlib`.

## 13. Ordinea de lucru

1. Pasul 0: inspectarea datelor și `docs/data_notes.md`.
2. Ținta din `busy`, cu raportul de calitate a datelor.
3. Feature-urile de stație și de competiție (doar din UrbanEV).
4. Split, baseline-urile B0 și B1, primul model LightGBM. **Fluxul complet funcțional înainte de rafinări.**
5. Feature-urile POI și populație.
6. Evaluarea completă, analiza erorilor, SHAP.
7. ICE și verificarea saturării.
8. README, curățarea codului, verificarea reproductibilității de la zero.

Faceți commit după fiecare pas.

## 14. Limite de menționat în prezentare

- Ținta e cererea **servită**, nu cererea totală.
- UrbanEV conține stațiile de pe o singură platformă, deci competiția poate fi subestimată.
- WorldPop e din 2020, iar datele de încărcare sunt din 2022–2023.
- Doar 6 luni (toamnă și iarnă), fără sezonalitate anuală.
- Datele de antrenare provin din stații construite deja de operatori în locuri pe care ei le-au considerat bune (selection bias).
