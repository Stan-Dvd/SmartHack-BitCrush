# Cererea la stațiile de încărcare EV: model ML (faza 1)

Smarthack XI 2026, tema PPC „24h to Power PPC”.

Modelul prezice **cererea de încărcare** la o stație: numărul mediu de puncte ocupate, pentru fiecare oră a zilei și fiecare tip de zi (lucrătoare sau weekend). Folosește doar informații care există și pentru o **locație nouă**: competiția din jur, populația, POI, mărimea și puterea stației, ora. Nu folosește coordonate, ID-uri, prețuri sau istoricul de ocupare. A fost antrenat pe datele UrbanEV din Shenzhen (1.406 stații, septembrie 2022 – februarie 2023) și va fi folosit ulterior de un agent care recomandă unde să se construiască stații noi.

Planul complet: [PLAN.md](PLAN.md). Note despre date: [docs/data_notes.md](docs/data_notes.md). Experimente și analiza erorilor: [docs/experiments.md](docs/experiments.md).

## Rezultate pe setul de test

Testul conține 20% din zonele de trafic (TAZ), alese o singură dată cu seed 42. Toate stațiile unei zone sunt fie în train, fie în test.

| Model | MAE (rând) | RMSE (rând) | R² (rând) | MAE ore încărcare/zi (stație) | Spearman cerere zilnică |
|---|---|---|---|---|---|
| B0 (profil oraș) | 3.269 | 5.334 | 0.008 | 74.26 | — |
| B1 (ocupare × n_piles) | 1.930 | 3.400 | 0.597 | 41.59 | 0.775 |
| **LightGBM** | **1.795** | **2.949** | **0.697** | **39.05** | 0.768 |

Set de test: 14.576 rânduri, 304 stații, 61 de zone (metricile pe stație folosesc 302 stații, cele cu toate cele 48 de celule). Media lui y în test e 3,757 puncte ocupate, adică 87,8 ore de încărcare pe zi per stație.

- **Rând** = stație × oră × tip de zi. **Ore de încărcare pe zi** = (5·Σ zi lucrătoare + 2·Σ weekend) / 7.
- **B0**: profilul mediu orar al orașului. **B1**: rata medie de ocupare per oră și tip de zi × numărul de puncte al stației. Ambele sunt calculate doar pe train.
- Față de B1, LightGBM reduce MAE cu 7% și RMSE cu 13% și crește R² de la 0,60 la 0,70. În **CV pe zone** (doar train), câștigul de MAE față de B1 e de 10,4%; pe blocuri spațiale mai mari (KMeans), de 8,3%.
- La **clasarea** stațiilor după cererea zilnică, modelul e la egalitate cu B1 (Spearman 0,768 față de 0,775). Clasarea stațiilor existente e dominată de mărimea lor. Avantajul modelului e în nivelul cererii, adică în cum modulează locația rata de ocupare.

### Ce contează (SHAP, grupat)

| Grup de feature-uri | medie \|Σ SHAP\| |
|---|---|
| competiție (`dist_nearest_station_m`, `n_stations_500m/1km/3km`, `competitor_piles_1km`) | 0,129 |
| stație (`n_piles`, `power_kw`) | 0,105 |
| populație și POI | 0,085 |
| timp (`hour`, `is_weekend`) | 0,078 |

Modelul prezice rata de ocupare, iar `n_piles` intră și ca multiplicator, deci efectul structural al mărimii nu apare în SHAP. Mai multe stații pe 3 km înseamnă ocupare mai mică per stație. Populația mare, puterea mare și orele de noapte cresc ocuparea. Figurile sunt în [reports/figures/](reports/figures/).

### Unde greșește

Detaliile sunt în [docs/experiments.md](docs/experiments.md#analiza-erorilor-set-de-test-modelul-final).

- **Stații mari, lente, periferice, cu încărcare de noapte** (depouri de flotă sau parcări rezidențiale): sunt subestimate, cu ocupare reală de 0,44 față de 0,28 prezisă. Ele produc 28% din eroarea totală prin doar 20 de stații. Tipul de utilizator nu se vede în niciun feature.
- **Noaptea** modelul subestimează sistematic (bias de −0,3 până la −0,6 puncte).
- **Stații „moarte”** (cerere aproape zero): sunt supraestimate.
- **Saturare:** modelul prezice medii și nu atinge pragul de ocupare de 0,9 la ora de vârf, deci precizia și recall-ul la prag sunt 0. Ca clasare are AUC 0,74 față de ocuparea reală de vârf peste 0,9.
- **`n_piles`:** cererea prezisă crește cu numărul de puncte, dar **super-liniar**, din cauza selection bias-ului (stațiile mari sunt construite unde cererea e deja mare). Agentul nu trebuie să aleagă `n_piles` doar după acest model.

### Încercări de îmbunătățire

După faza 1 am testat în CV 22 de candidați: stațiile cu acoperire 50–70%, `fast_share`, populația și POI pe alte raze, competiția ponderată cu distanța, rapoarte cerere/ofertă și 10 feature-uri OpenStreetMap despre tipul locului (landuse, drumuri mari, parcări, gări, clădiri). **Niciunul nu a trecut regula de adoptare**: mai bun în medie și în cel puțin 4 din 5 fold-uri.

Un diagnostic out-of-fold arată că eroarea rămasă a modelului nu e corelată cu niciun feature disponibil (|ρ| ≤ 0,06), dar e corelată cu **prețul** stației (ρ = −0,22), exclus intenționat din model. Detalii în [docs/experiments.md](docs/experiments.md#îmbunătățiri-încercate-după-faza-1-niciuna-adoptată).

## Rulare

Cerințe: Python 3.13 (testat cu 3.13.9), aproximativ 6 GB de date brute și 4,5 GB temporar pentru WorldPop.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.download        # verifică UrbanEV; descarcă WorldPop (~4,4 GB) și OSM Guangdong (~175 MB), o singură dată
python -m src.inspect_data    # pasul 0: structura datelor (docs/data_notes.md)
python -m src.build_dataset   # ținta + feature-urile -> data/processed/train_table.parquet
python -m src.split           # split pe zone -> data/processed/split.json (NU suprascrie fișierul existent)
python -m src.train           # CV pe zone + modelul final -> models/lgbm.txt, models/params.json
python -m src.evaluate        # B0, B1, LightGBM pe test -> reports/metrics.json, reports/metrics_table.md
python -m src.analysis        # erori, SHAP, ICE/PD, saturare -> reports/figures/, reports/analysis.json
python -m src.experiments     # ablații și verificări de robustețe (doar CV) -> reports/experiments.json
python -m src.experiments --candidates   # feature-urile candidate (doar CV) -> reports/experiments_candidates.json
```

Tot fluxul durează aproximativ 3 minute pe 16 nuclee, fără descărcarea WorldPop.

**Date UrbanEV.** Dryad nu permite o descărcare automată fiabilă, așa că `src.download` doar verifică dacă fișierele există și, dacă lipsesc, afișează instrucțiuni. Descărcați datele brute pe stații de pe Dryad și dezarhivați-le în `data/raw/UrbanEVDataset/20220901-20230228_station-raw/`. Datele suplimentare (POI) vin din repository-ul GitHub UrbanEV și se pun în `data/raw/UrbanEVSupplemental/`.

**Reproductibilitate.**
- `split.json` e în repository și e sursa de adevăr pentru split. A fost generat înainte de regula care elimină seriile blocate, iar regenerarea lui (`--force`) ar alege alte zone de test.
- LightGBM rulează cu `deterministic=True` și `seed=42`, deci două rulări consecutive dau același `metrics.json`.
- Toate constantele (raze, praguri, seed) sunt în [src/config.py](src/config.py).

## Ținta și feature-urile

- **Ținta y:** numărul mediu de puncte ocupate (`busy`) per stație × oră × tip de zi, calculat doar pe intervalele disponibile de 5 minute.
- **Stații eliminate:**
  - 21 fără date;
  - 238 cu acoperire sub 70%;
  - 17 cu serii blocate pe plin (toate punctele ocupate în peste 95% din timp, un artefact de raportare).
- **Celule eliminate:** 38 de celule cu mai puțin de 10 observații. Rămân 1.406 stații și 67.381 de rânduri ([reports/data_quality.json](reports/data_quality.json)).
- **Modelul:** LightGBM (tweedie 1,2, 31 de frunze, `min_child_samples = 1000`, 108 runde). Prezice **rata de ocupare** y/n_piles, cu pondere n_piles, iar predicția se înmulțește cu n_piles. Ocuparea e constrânsă să nu scadă când crește `power_kw`. Alegerea s-a făcut prin CV GroupKFold(5) pe zone, doar în train.

| Feature | Definiție |
|---|---|
| `dist_nearest_station_m` | distanța haversine până la cea mai apropiată altă stație (din toate cele 1.682 de stații brute) |
| `n_stations_500m`, `n_stations_1km`, `n_stations_3km` | numărul altor stații pe rază |
| `competitor_piles_1km` | numărul de puncte ale celorlalte stații pe 1 km |
| `pop_1km` | populația WorldPop 2020 (100 m) pe 1 km |
| `poi_food_500m`, `poi_business_res_500m`, `poi_lifestyle_500m` | numărul de POI pe 500 m, pe categorii |
| `n_piles` | numărul de puncte de încărcare |
| `power_kw` | puterea nominală medie pe punct; pentru cele 130 de stații fără date de putere, imputată din numărul de puncte AC și DC |
| `hour`, `is_weekend` | ora 0–23 și sâmbătă/duminică |

**Excluse intenționat:** coordonate, ID-uri de stație și zonă, orice derivat din `busy`, prețul, vremea și sezonul (vezi PLAN.md §6.4 și §8).

## Structura

```
src/config.py            constante: căi, raze, praguri, seed, lista de feature-uri
src/download.py          verificarea UrbanEV + WorldPop (descărcare și decupaj Shenzhen)
src/inspect_data.py      pasul 0
src/load_urbanev.py      citirea fișierelor UrbanEV și POI
src/target.py            busy -> y, filtre, data_quality.json
src/features_*.py        feature-uri spațiale (haversine, BallTree), de stație, temporale
src/build_dataset.py     -> train_table.parquet, station_features.parquet
src/split.py             split pe zone (seed 42)
src/baselines.py         B0, B1
src/train.py             CV pe zone, grid, modelul final
src/evaluate.py          metrici pe test
src/analysis.py          erori, SHAP, ICE/PD, saturare
src/experiments.py       ablații și robustețe (doar CV)
docs/                    note despre date, jurnalul experimentelor
models/                  lgbm.txt, params.json
reports/                 metrici, analize, figuri
```

## Limite

- Ținta e **cererea servită**. La stațiile pline, cererea reală e mai mare decât cea observată.
- UrbanEV conține stațiile unei singure platforme, deci competiția poate fi subestimată.
- WorldPop e din 2020, iar datele de încărcare din 2022–2023.
- Datele acoperă doar 6 luni (toamnă și iarnă), fără sezonalitate anuală. Sărbătorile și zilele lucrătoare mutate din China nu sunt tratate separat.
- **Selection bias:** stațiile au fost construite de operatori în locuri pe care le-au considerat bune, iar mărimea lor a fost aleasă după cererea așteptată. Efectul învățat al lui `n_piles` nu e cauzal.
- Zonele TAZ sunt mici, deci o parte din câștig poate veni din autocorelarea spațială. Pe blocuri spațiale mai mari, câștigul în CV față de B1 scade de la 10,4% la 8,3%.

## Surse de date și licențe

- **UrbanEV** (CC0 1.0). Li, H., Qu, H., Tan, X. et al. *UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction.* Scientific Data 12, 523 (2025). https://doi.org/10.1038/s41597-025-04874-4. Datele brute pe stații: https://doi.org/10.5061/dryad.np5hqc04z. Repository: https://github.com/IntelligentSystemsLab/UrbanEV
- **OpenStreetMap** (ODbL), © OpenStreetMap contributors, extractul Guangdong de la Geofabrik (https://download.geofabrik.de). Folosit doar pentru feature-urile candidate.
- **WorldPop** (CC BY 4.0), China 2020, 100 m, ajustat la UN (`chn_ppp_2020_UNadj.tif`). WorldPop (www.worldpop.org - School of Geography and Environmental Science, University of Southampton; Department of Geography and Geosciences, University of Louisville; Departement de Geographie, Universite de Namur) and Center for International Earth Science Information Network (CIESIN), Columbia University (2018). Global High Resolution Population Denominators Project - Funded by The Bill and Melinda Gates Foundation (OPP1134076). https://dx.doi.org/10.5258/SOTON/WP00660
