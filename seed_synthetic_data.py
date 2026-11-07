"""
Generează date demonstrative pentru UrbanEV în PostgreSQL.

Datele sunt sintetice și sunt marcate prin intervalul de ID-uri 900001+.
Rulare:
    python seed_synthetic_data.py
    python seed_synthetic_data.py --stations 50 --hours 72 --seed 7

Scriptul este idempotent pentru intervalul de ID-uri folosit: șterge observațiile
stațiilor sintetice din acel interval și le regenerează, fără să atingă stațiile reale.
"""
import argparse
import math
import random
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.db import open_pool, close_pool, get_connection
from app.h3_utils import station_cells
from app.repository import ensure_h3_cell


# Coordonate aproximative ale unor zone din București; punctele generate în jurul
# lor sunt DEMONSTRATIVE, nu reprezintă adrese sau stații reale.
AREAS = [
    ("Centru", 44.4325, 26.1039),
    ("Piața Victoriei", 44.4448, 26.0960),
    ("Universitate", 44.4355, 26.1020),
    ("Tineretului", 44.4115, 26.1050),
    ("Grozăvești", 44.4390, 26.0600),
    ("Drumul Taberei", 44.4160, 26.0350),
    ("Titan", 44.4210, 26.1690),
    ("Obor", 44.4490, 26.1280),
    ("Băneasa", 44.5060, 26.0880),
    ("Vitan", 44.4080, 26.1320),
    ("Militari", 44.4340, 26.0180),
    ("Pipera", 44.5020, 26.1250),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def demand_factor(ts: datetime, area_index: int) -> float:
    """Cerere zilnică sintetică: vârf dimineața și seara, variație între zone."""
    hour = ts.hour + ts.minute / 60
    morning = math.exp(-((hour - 8.2) / 2.0) ** 2)
    evening = math.exp(-((hour - 18.3) / 2.7) ** 2)
    daytime = 0.18 + 0.30 * morning + 0.48 * evening
    # Unele zone comerciale au activitate mai mare în timpul zilei.
    commercial = 0.12 * math.exp(-((hour - 13.0) / 3.0) ** 2) if area_index % 3 == 0 else 0
    return clamp(daytime + commercial, 0.08, 0.95)


def main():
    parser = argparse.ArgumentParser(description="Seed date sintetice UrbanEV")
    parser.add_argument("--stations", type=int, default=36)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-id", type=int, default=900001)
    parser.add_argument("--pump-rating-kw", type=float, default=None)
    args = parser.parse_args()

    if args.stations < 1 or args.hours < 1:
        parser.error("--stations și --hours trebuie să fie pozitive")

    rng = random.Random(args.seed)
    rating = args.pump_rating_kw or settings.pump_rating_kw
    interval = settings.snapshot_minutes
    now = datetime.now(timezone.utc)
    end = now.replace(minute=(now.minute // interval) * interval, second=0, microsecond=0)
    start = end - timedelta(hours=args.hours)
    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current)
        current += timedelta(minutes=interval)

    stations = []
    for i in range(args.stations):
        area_index = i % len(AREAS)
        area_name, center_lat, center_lon = AREAS[area_index]
        # Dispersie locală mică pentru o distribuție vizibilă pe harta Bucureștiului.
        lat = center_lat + rng.uniform(-0.009, 0.009)
        lon = center_lon + rng.uniform(-0.012, 0.012)
        charge_count = rng.choice([2, 4, 4, 6, 8])
        station_id = args.start_id + i
        h3_8, h3_9 = station_cells(lat, lon)
        stations.append({
            "id": station_id, "lat": lat, "lon": lon,
            "charge_count": charge_count, "taz_id": area_index + 1,
            "area": rng.uniform(0.5, 3.5), "perimeter": rng.uniform(3.0, 9.0),
            "h3_8": h3_8, "h3_9": h3_9, "area_name": area_name,
            "base": rng.uniform(0.45, 1.15),
        })

    open_pool()
    try:
        with get_connection() as conn:
            with conn.transaction():
                # Se creează geometria celulelor H3 folosite de stațiile demonstrative.
                seen = set()
                for st in stations:
                    for cell, res in ((st["h3_8"], 8), (st["h3_9"], 9)):
                        if cell not in seen:
                            ensure_h3_cell(conn, cell, res)
                            seen.add(cell)

                # Upsert doar pentru ID-urile sintetice rezervate.
                for st in stations:
                    conn.execute("""
                        INSERT INTO statie_incarcare
                            (statie_id, locatie, h3_res8, h3_res9, charge_count,
                             taz_id, area, perimeter)
                        VALUES (
                            %s, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,
                            %s,%s,%s,%s,%s,%s
                        )
                        ON CONFLICT (statie_id) DO UPDATE SET
                            locatie = EXCLUDED.locatie,
                            h3_res8 = EXCLUDED.h3_res8,
                            h3_res9 = EXCLUDED.h3_res9,
                            charge_count = EXCLUDED.charge_count,
                            taz_id = EXCLUDED.taz_id,
                            area = EXCLUDED.area,
                            perimeter = EXCLUDED.perimeter
                    """, [st["id"], st["lon"], st["lat"], st["h3_8"], st["h3_9"],
                          st["charge_count"], st["taz_id"], st["area"], st["perimeter"]])

                # Curățăm doar observațiile generate de acest script.
                ids = [st["id"] for st in stations]
                conn.execute(
                    "DELETE FROM date_temporale_statie WHERE statie_id = ANY(%s)",
                    [ids],
                )

                # Inserarea folosește partițiile lunare existente sau partiția DEFAULT
                # definită în schema SQL; astfel nu mutăm/alterăm observații deja existente.

                rows = []
                for st in stations:
                    for ts in timestamps:
                        factor = demand_factor(ts, st["taz_id"] - 1) * st["base"]
                        # Ocupare aleatoare, dar corelată cu ora zilei și capacitatea stației.
                        probability = clamp(factor * 0.78, 0.03, 0.88)
                        busy = sum(1 for _ in range(st["charge_count"])
                                   if rng.random() < probability)
                        # Prețuri sintetice plauzibile, cu variație redusă în timp.
                        e_price = round(1.25 + 0.12 * math.sin((ts.hour / 24) * 2 * math.pi)
                                        + rng.uniform(-0.035, 0.035), 4)
                        s_price = round(0.42 + 0.035 * math.cos((ts.hour / 24) * 2 * math.pi)
                                        + rng.uniform(-0.015, 0.015), 4)
                        volume_kwh = busy * rating * interval / 60.0
                        rows.append((st["id"], ts, busy, round(volume_kwh, 6),
                                     max(0.01, e_price), max(0.01, s_price)))

                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO date_temporale_statie
                            (statie_id, timestamp, busy, volume, e_price, s_price)
                        VALUES (%s,%s,%s,%s,%s,%s)
                    """, rows)

        print("Date sintetice inserate cu succes.")
        print(f"Stații: {len(stations)} (ID {args.start_id}–{args.start_id + len(stations) - 1})")
        print(f"Interval: {start.isoformat()} – {end.isoformat()} UTC")
        print(f"Snapshot: {interval} minute | Observații: {len(stations) * len(timestamps):,}")
        print(f"Putere/pompă: {rating} kW | Energie calculată în kWh")
        print("Datele sunt demonstrative, nu reprezintă utilizare reală.")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
