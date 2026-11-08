# Note despre date (pasul 0)

Generat cu `python -m src.inspect_data` pe datele UrbanEV brute la nivel de stație (`data/raw/UrbanEVDataset/20220901-20230228_station-raw/`) și pe POI-urile suplimentare (`data/raw/UrbanEVSupplemental/20221201-shenzhen-poi.csv`).

## Fișiere

| Fișier | Rânduri | Conținut |
|---|---|---|
| `station_information.csv` | 1.682 | `station_id, longitude, latitude, slow_count, fast_count, charge_count, TAZID`, fără lipsuri |
| `pile_rated_power.csv` | 22.650 | `pile_id, power (kW), pileType (AC/DC), station_id`, fără lipsuri |
| `station_distance.csv` | 1.682 × 1.682 | matrice de distanțe în metri; prima coloană `ID`, header-ul conține ID-urile stațiilor |
| `charge_5min/{station_id}.csv` | 1.682 fișiere × 52.128 | `time, busy, idle, s_price, e_price, fast_busy, fast_idle, slow_busy, slow_idle, duration, volume` |
| `20221201-shenzhen-poi.csv` | 712.135 | `primary_types, longitude, latitude` |

## Răspunsuri

**1. Ce este `busy`?** Numărul de puncte ocupate **per stație**, în format **lung**: un fișier per stație, cu un rând la 5 minute, de la 2022-09-01 00:00 până la 2023-02-28 23:55 (52.128 de intervale). Suma `busy + idle` e constantă în timp pentru toate stațiile cu date și e egală cu `charge_count` în 100% din cazuri. Intervalele lipsă apar ca rânduri cu toate valorile goale.

**2. Puterea și numărul de puncte.** Numărul de puncte e `charge_count` din `station_information.csv` (= `slow_count + fast_count`). Puterea e dată **per punct**, în `pile_rated_power.csv`, și trebuie agregată: `power_kw` = media pe punctele stației. Probleme:
- 82 de puncte au putere 0 și sunt tratate ca lipsă;
- **130 de stații nu au niciun punct** în fișier, iar pentru ele puterea se imputează ca `(slow_count·median_AC + fast_count·median_DC) / charge_count` (median AC = 7 kW);
- la 7% dintre stațiile prezente, numărul de puncte din fișier diferă de `charge_count`. Folosim `charge_count`, care coincide cu `busy + idle`.

**3. Zona.** Da: `TAZID` (traffic analysis zone), cu 331 de zone unice și fără lipsuri. Se folosește pentru split.

**4. POI.** **Puncte cu coordonate** (WGS84), în 3 categorii: `lifestyle services` (393.428), `business and residential` (183.383), `food and beverage services` (135.324). Feature-urile POI sunt deci numărători pe o rază de 500 m.

**5. ID-uri.** `station_id` e întreg și e identic între `station_information`, numele fișierelor din `charge_5min/` și matricea de distanțe. În `pile_rated_power` ID-ul e stocat ca float și se convertește la int. ID-urile din fișierul de puncte sunt o submulțime a celor din `station_information`.

## Alte observații

- **Acoperire** (fracțiunea de intervale cu date): stații cu acoperire ≥30%: 1.543; ≥50%: 1.511; **≥70%: 1.423**; ≥90%: 836. 21 de stații nu au nicio observație. Curățarea autorilor UrbanEV păstrează 1.362 de stații, cu acoperirea minimă 70,07%. Folosim **pragul de 70%**.
- 74 de stații încep abia pe 2023-02-01, iar 578 de stații se opresc pe 2023-02-21, în jurul orei 08:25 (date trunchiate la sursă).
- **Saturare:** în medie, 4,7% din intervale au toate punctele ocupate (mediana e 0,3%).
- **264 de stații** au coordonate identice cu cel puțin o altă stație: situri împărțite de operator, de exemplu în puncte lente și rapide. Pentru ele `dist_nearest_station_m = 0`, iar stația de pe același sit apare în `competitor_piles_1km`. Le păstrăm așa, pentru că sunt infrastructură reală la acel loc.
- **Distanțe:** `station_distance.csv` e distanță în linie dreaptă. Față de haversine diferă cu 0,14% (mediană) și maximum 161 m pe distanțe de zeci de km, ceea ce înseamnă un elipsoid diferit. Folosim **haversine** pentru toate feature-urile pe raze, pentru că agentul va calcula la fel pentru locațiile noi.
- Bounding box-ul stațiilor: lon 113,78–114,52, lat 22,47–22,82. Pentru WorldPop decupăm lon 113,70–114,70, lat 22,35–22,90.

## WorldPop

- `chn_ppp_2020_UNadj.tif` (100 m, persoane per pixel, ajustat la UN), descărcat de pe data.worldpop.org.
- Serverul nu acceptă cereri pe intervale de bytes, deci citirea pe fereastră prin `/vsicurl/` eșuează. `src/download.py` descarcă fișierul complet (~4,4 GB) și îl decupează pe bounding box-ul Shenzhen.
- `pop_1km` = suma pixelilor cu centrul la cel mult 1 km de stație.
