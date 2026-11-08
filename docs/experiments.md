# Jurnalul experimentelor

## Cum s-a evaluat

- Toate alegerile de model (mod țintă, obiectiv, hiperparametri, feature-uri, constrângeri) s-au făcut prin **CV GroupKFold(5) pe zone, doar în train**: 1.102 stații, 253 de zone, 52.805 rânduri. MAE e calculat mereu pe `y` (puncte ocupate).
- Comparațiile din `src/experiments.py` folosesc **aceleași fold-uri**. Pentru fiecare experiment raportăm diferența medie față de referință (Δ) și în câte din cele 5 fold-uri e mai bun. Cu doar 5 fold-uri, o diferență sub ~0,01 e zgomot.
- **Igiena setului de test.** Testul a fost rulat de câteva ori în timpul dezvoltării, ca verificare a fluxului cap-coadă, dar nicio alegere de hiperparametri, țintă sau feature-uri nu s-a făcut după scorul de test. O singură decizie a pornit de la analiza pe test: regula pentru seriile blocate (punctul 2 de mai jos). E o regulă de calitate a datelor, aplicată uniform tuturor stațiilor înainte de split. `split.json` **nu** a fost regenerat după ea: zonele de test au rămas cele alese inițial cu seed 42.

Pentru reproducere: `python -m src.train`, `python -m src.experiments` (aproximativ 30 s), cu rezultatele în `reports/cv_results.csv` și `reports/experiments.json`.

## Cronologie

| # | Ce s-a schimbat | CV MAE | B1 CV | Câștig vs B1 | Observații |
|---|---|---|---|---|---|
| 1 | Flux de bază: ținta `y`, fără POI/populație, grid inițial (15/31/63 frunze, min_child 50/200), L2 vs tweedie 1,2/1,5 | 1,782 | 1,876 | +5,0% | Tweedie 1,5 > tweedie 1,2 > L2 la fiecare combinație. Pe test: MAE 1,865 față de 2,035 la B1, R² 0,681 față de 0,561. |
| 2 | **Regulă de calitate:** eliminate 17 stații „blocate pe plin” (`sat_share > 0,95`) | 1,725 | 1,854 | +6,9% | Găsite în profilurile din analiza erorilor. Exemplu: stația 1265 are 45/45 puncte ocupate în 99,97% din intervale timp de 6 luni, iar 4 stații au `busy` perfect constant. Rămân ca infrastructură în feature-urile de competiție. |
| 3 | + POI și `pop_1km` (feature-urile din plan) | 1,755 | 1,854 | +5,3% | **Mai rău** decât fără ele. Cea mai bună configurație era la marginea grilei (cei mai mici arbori), deci overfitting pe locație. |
| 4 | Grilă extinsă spre regularizare: 4–31 frunze, `min_child_samples` 200/500/1000 | 1,732 | 1,854 | +6,5% | Cele 48 de rânduri ale unei stații sunt puternic corelate, deci există doar aproximativ 1.100 de exemple independente. 1.000 de rânduri per frunză înseamnă cam 20 de stații. |
| 5 | **Ținta = rata de ocupare `y/n_piles`**, cu pondere `n_piles`, iar predicția se înmulțește cu `n_piles` | 1,669 | 1,854 | +10,0% | Structura lui B1 (ocupare × mărime) e impusă, iar modelul învață doar cum modulează locația ocuparea. Cu ponderea `n_piles`, L1 pe ocupare e proporțional cu MAE pe `y`. Cu această formulare **toate** grupurile de feature-uri ajută (tabelul de mai jos). |
| 6 | Constrângere monotonă: ocupare nedescrescătoare în `power_kw` (**modelul final**) | **1,662** | 1,854 | **+10,4%** | Fără ea, PD pe `power_kw` avea o mică scădere la 6–7 kW, iar doar 4% din curbele ICE erau monotone. Cu ea, 100% sunt monotone, fără cost în CV. |

Modelul final: `target_mode = occ`, `objective = tweedie (1,2)`, `num_leaves = 31`, `min_child_samples = 1000`, `learning_rate = 0,05`, `feature_fraction = 0,8`, 108 runde, constrângere `power_kw: +1`. Detalii în `models/params.json`.

## Experimente față de modelul final (CV pe zone, aceleași fold-uri)

| Experiment | CV MAE | Δ față de final | Fold-uri mai bune | Câștig vs B1 |
|---|---|---|---|---|
| **modelul final** (ocupare, tweedie 1,2, monoton pe `power_kw`) | **1,662** | — | — | **+10,4%** |
| fără populație și POI | 1,694 | +0,032 | 1/5 | +8,6% |
| fără cele 3 feature-uri POI | 1,669 | +0,008 | 1/5 | +9,9% |
| fără `pop_1km` | 1,688 | +0,026 | 1/5 | +8,9% |
| fără feature-urile de competiție | 1,723 | +0,061 | 0/5 | +7,1% |
| fără `power_kw` | 1,689 | +0,028 | 1/5 | +8,9% |
| ținta `y` directă, tweedie 1,2 (abordarea inițială) | 1,776 | +0,114 | 0/5 | +4,2% |
| ținta `y` directă, L2 | 1,799 | +0,138 | 0/5 | +2,9% |
| ocupare, L2 | 1,688 | +0,026 | 0/5 | +9,0% |
| ocupare, cross-entropy | 1,676 | +0,014 | 1/5 | +9,6% |
| `learning_rate` 0,02 | 1,674 | +0,012 | 2/5 | +9,7% |
| `lambda_l2` = 10 | 1,675 | +0,014 | 0/5 | +9,6% |
| fără constrângeri monotone | 1,675 | +0,014 | 2/5 | +9,6% |
| ocupare **necrescătoare** în `n_piles` (randament descrescător) | 1,765 | +0,104 | 0/5 | +4,8% |
| + `fast_share` (ponderea punctelor DC; **nu** e în model) | 1,658 | −0,003 | 3/5 | +10,5% |
| CV pe 50 de blocuri spațiale KMeans în loc de zone TAZ | 1,705 | n/a | n/a | +8,3% |

Ce reiese:
- **Competiția** e cel mai important grup: fără ea modelul e mai slab în toate cele 5 fold-uri. Urmează populația și `power_kw`. POI-urile aduc puțin (+0,008), dar constant, iar cele trei categorii sunt foarte corelate între ele (Spearman 0,75–0,94).
- **Formularea țintei** contează mai mult decât orice hiperparametru: de la 4,2% la 10,4% câștig față de B1.
- **Robustețe spațială.** Zonele TAZ sunt mici (aproximativ 1,5 km²), deci stații vecine pot ajunge în train și validare, iar câștigul ar putea fi umflat de autocorelarea spațială. Pe blocuri KMeans mult mai mari, câștigul scade de la 10,4% la 8,3%, dar rămâne. Prin urmare, cifrele pe zone TAZ sunt ușor optimiste.

## Ce s-a încercat pentru corectarea erorilor și cu ce rezultat

1. **Serii blocate pe plin** (eroarea cea mai mare din primul model). Eliminarea celor 17 stații a crescut câștigul în CV față de B1 de la +5,0% la +6,9%.
2. **Overfitting pe locație** (POI/populația înrăutățeau CV-ul). Grila extinsă spre regularizare și, mai ales, ținta ca rată de ocupare au rezolvat problema: acum feature-urile de context ajută.
3. **Efectul lui `n_piles` nu e „tot mai încet”.** Cererea prezisă crește cu `n_piles`: PD crescător, 93% din curbele ICE nedescrescătoare. Dar crește **convex**: panta e 0,22 puncte ocupate per punct adăugat în jumătatea de jos și 0,39 în cea de sus, iar ocuparea prezisă per punct urcă de la 0,22 (1–20 de puncte) la 0,36 (66 de puncte). Cauza e selection bias: operatorii construiesc stații mari unde cererea e deja mare. Am încercat constrângerea „ocupare necrescătoare în `n_piles`”. Ea face cererea **exact liniară** în `n_piles` (randament constant), costă +0,104 MAE în CV și pierde jumătate din câștig. O constrângere monotonă nu poate produce concavitate, iar datele transversale nu identifică randamentul descrescător. **Nu am adoptat-o.** Consecința pentru agent e descrisă mai jos.
4. **`power_kw` nemonoton.** Corectat cu constrângere monotonă, fără cost (punctul 6 din cronologie).

## Analiza erorilor (set de test, modelul final)

Fișiere: `reports/analysis.json`, `reports/top20_stations.csv` și `reports/figures/`.

- **Pe mărime:** eroarea absolută crește cu `n_piles` (MAE 0,61 în Q1 față de 3,93 în Q4), dar eroarea relativă scade (69% față de 41%). Câștigul față de B1 vine aproape integral din stațiile mari (Q4: 3,93 față de 4,45). La stațiile mici, modelul și B1 sunt la egalitate.
- **Pe context:** câștigul cel mai mare apare în zonele cu populație medie spre mare (`pop_1km` Q3: 1,31 față de 1,54) și cu densitate medie de stații (`n_stations_1km` Q2–Q3). În zonele cele mai dense (Q4) modelul e chiar puțin mai slab decât B1 (1,16 față de 1,10).
- **Pe oră:** modelul **subestimează sistematic noaptea** (bias de aproximativ −0,3 până la −0,6 puncte între 21:00 și 06:00) și e aproape nedeplasat ziua. Eroarea e maximă noaptea, atât la B1, cât și la LightGBM (`residuals_by_hour.png`).
- **Top 20 de stații** (28% din eroarea absolută totală, 14 subestimate și 6 supraestimate): **mari** (mediana 35,5 puncte, față de 10 în test), **lente** (AC 7 kW), **periferice** (`n_stations_3km` 30 față de 59, `pop_1km` 27k față de 45k, POI de 2–4 ori mai puține). Ocuparea lor reală e 0,44, iar cea prezisă 0,28. Profilurile lor orare (`profiles_typical_atypical.png`) arată încărcare masivă noaptea și ocupare scăzută ziua: depouri de flotă (taxi, ride-hailing, logistică) sau parcări rezidențiale. **Tipul de utilizator al stației nu se vede în niciun feature**, iar aceasta e principala sursă de eroare.
- **Eroare relativă mare:** stații cu cerere aproape zero (ocupare reală 4%, prezisă 23%), toate supraestimate. Sunt stații „moarte”, pe care modelul nu le poate distinge de vecinele lor.
- **Saturare:** modelul prezice medii orare și se contractă spre medie (ocuparea prezisă la ora de vârf are maximum 0,65, iar cea reală 0,94), deci **nu atinge niciodată pragul de 0,9**. Precizia și recall-ul la prag sunt nedefinite sau 0. Ca clasare, stațiile cu cea mai mare ocupare de vârf prezisă conțin 2× (față de `sat_share ≥ 10%`) și 19× (față de ocuparea reală de vârf > 0,9) mai multe stații saturate decât rata de bază. AUC e 0,57, respectiv 0,74. Saturarea nu poate fi citită direct dintr-un prag pe predicțiile acestui model.

## Concluzii pentru faza următoare (agentul)

- **Nu lăsați agentul să aleagă `n_piles` doar după acest model.** Răspunsul învățat e super-liniar și reflectă unde au construit operatorii stații mari, nu efectul cauzal al adăugării de puncte. Variante: ocuparea evaluată la mărimea medie a stațiilor din zonă, plafonată, sau o logică explicită de capacitate (cerere în vehicule și servire limitată de numărul de puncte).
- `power_kw` are acum un efect monoton (ocupare mai mare la putere mai mare) și poate fi folosit ca variabilă de decizie, cu aceeași prudență legată de selection bias.
- Câștiguri probabile în acuratețe: tipul de stație sau utilizatorul (public, flotă, rezidențial), dacă există o sursă publică; `fast_share` (câștig marginal, 3/5 fold-uri).
- Pentru clasarea locațiilor, LightGBM e la egalitate cu B1 (Spearman 0,768 față de 0,775 pe test). Avantajul modelului e în **nivelul** cererii (RMSE −13%), nu în ordonarea stațiilor existente, care e dominată de mărimea lor.

## Îmbunătățiri încercate după faza 1 (niciuna adoptată)

**Scop:** reducerea erorilor descrise mai sus (stații de flotă sau depou, subestimarea de noapte, stații „moarte”). Am urmărit trei direcții alese împreună cu echipa:
1. date de antrenare în plus;
2. feature-uri pe mai multe raze;
3. feature-uri OpenStreetMap despre tipul locului.

**Regula de adoptare**, fixată dinainte: un candidat intră în model doar dacă îmbunătățește media CV **și** e mai bun în cel puțin 4 din 5 fold-uri, comparat pe aceleași fold-uri cu modelul final (CV MAE 1,6616). Testul **nu** a fost rulat pentru candidați. Rezultatele sunt în `reports/experiments_candidates.json` și se reproduc cu `python -m src.experiments --candidates`.

| Candidat | CV MAE | Δ față de final | Fold-uri mai bune |
|---|---|---|---|
| + stațiile cu acoperire 50–70% (83 de stații, doar în train) | 1,687 | +0,026 | 1/5 |
| + `fast_share` | 1,658 | −0,003 | 3/5 |
| + `pop_500m` | 1,665 | +0,004 | 2/5 |
| + `pop_3km` | 1,660 | −0,001 | 3/5 |
| + POI pe 1 km (food / business_res / lifestyle, individual) | 1,675–1,693 | +0,013…+0,032 | 1–2/5 |
| + `competition_weighted` (Σ puncte · e^(−d/1 km)) | 1,665 | +0,003 | 4/5 |
| + `pop_per_pile_1km` | 1,675 | +0,014 | 3/5 |
| + `poi_per_pile_1km` | 1,669 | +0,007 | 2/5 |
| + tot grupul pe mai multe raze (8 feature-uri) | 1,687 | +0,026 | 1/5 |
| + OSM `landuse_residential/industrial/commercial_500m` (individual) | 1,666–1,684 | +0,004…+0,022 | 2–3/5 |
| + OSM `dist_major_road_m` | 1,677 | +0,015 | 1/5 |
| + OSM `major_road_km_1km` | 1,654 | −0,008 | 3/5 |
| + OSM `n_parking_500m` / `dist_parking_m` | 1,666 / 1,680 | +0,004 / +0,019 | 3/5 / 1/5 |
| + OSM `dist_rail_station_m` / `dist_bus_station_m` | 1,669 / 1,658 | +0,008 / −0,004 | 2/5 / 3/5 |
| + OSM `n_buildings_500m` | 1,669 | +0,007 | 2/5 |
| + tot grupul OSM (10 feature-uri) | 1,674 | +0,013 | 1/5 |

**Diagnostic: există semnal pe care îl ratăm?** Am calculat predicții out-of-fold pentru cele 1.102 stații din train și reziduul de ocupare pe stație (real minus prezis, împărțit la `n_piles`). Apoi am măsurat corelația Spearman a reziduului cu fiecare feature, atât candidat, cât și existent:
- **toate** corelațiile au |ρ| ≤ 0,06, adică sub pragul de zgomot pentru 1.102 stații (±0,06);
- la fel pentru reziduul de noapte.

Ce greșește modelul nu se explică prin densitate, rază, tipul terenului din OSM sau mărime. Feature-urile noi aduc doar posibilități de overfitting: cu ~1.100 de exemple independente, fiecare feature în plus costă mai mult decât aduce.

**Unde e informația lipsă.** Singura variabilă găsită care e corelată cu reziduul e **prețul**: tariful mediu pe kWh (`e_price`) are ρ = −0,19, iar prețul total (energie + serviciu) are ρ = −0,22. Stațiile ieftine sunt mai pline decât prezice modelul. Prețul rămâne exclus, conform planului (§6.4: o stație nouă nu are preț, iar efectul e greu de separat de cauzalitatea inversă). Pentru faza agentului, însă, prețul e o **variabilă de decizie** a operatorului, ca `n_piles`, și merită discutat ca input controlat de agent.

**Alte observații:**
- Acoperirea OSM în China e slabă: doar ~10% din grila Shenzhen are landuse marcat, iar clădirile sunt ~170 de mii, cu mult sub numărul real. În România OSM e mult mai complet, deci aceste feature-uri pot deveni utile acolo. Codul rămâne în `src/features_osm.py`.
- Stațiile cu acoperire 50–70% strică modelul: medii calculate pe mai puține zile sunt mai zgomotoase. Pragul de 70% e confirmat.
