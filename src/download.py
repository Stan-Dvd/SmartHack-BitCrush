"""Pregătește datele brute în data/raw/.

- UrbanEV: verifică dacă fișierele există. Dryad nu permite o descărcare automată fiabilă,
  așa că, dacă lipsesc, scriptul afișează instrucțiuni.
- WorldPop 2020 (100 m, China, UN-adjusted): citește prin HTTP doar fereastra Shenzhen
  (GDAL /vsicurl/ cu range requests). Dacă nu merge, descarcă tot fișierul (~4,4 GB) și îl decupează.
"""
import shutil
import sys
import urllib.request

import rasterio
from rasterio.windows import from_bounds

from src import config as C

URBANEV_REQUIRED = [
    C.STATION_INFO,
    C.PILE_POWER,
    C.STATION_DISTANCE,
    C.CHARGE_5MIN_DIR,
    C.POI_FILE,
]

URBANEV_HELP = f"""
Lipsesc fișiere UrbanEV. Descarcă manual:
  1. Date brute pe stații (Dryad, CC0): https://doi.org/10.5061/dryad.np5hqc04z
     -> dezarhivează în {C.RAW / 'UrbanEVDataset'} astfel încât să existe
        {C.URBANEV_RAW.relative_to(C.ROOT)}/{{station_information.csv, pile_rated_power.csv,
        station_distance.csv, charge_5min/}}
  2. Date suplimentare (GitHub, POI): https://github.com/IntelligentSystemsLab/UrbanEV
     -> copiază folderul de date suplimentare în {C.URBANEV_SUPP.relative_to(C.ROOT)}/
        (trebuie să conțină {C.POI_FILE.name})
"""


def check_urbanev() -> bool:
    missing = [p for p in URBANEV_REQUIRED if not p.exists()]
    if missing:
        print("[UrbanEV] lipsă:", *[f"  - {m.relative_to(C.ROOT)}" for m in missing], sep="\n")
        print(URBANEV_HELP)
        return False
    n = len(list(C.CHARGE_5MIN_DIR.glob("*.csv")))
    print(f"[UrbanEV] OK ({n} fișiere de serie în charge_5min/)")
    return True


def _write_crop(src, out_path):
    b = C.BBOX
    win = from_bounds(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"], src.transform)
    win = win.round_offsets().round_lengths()
    data = src.read(1, window=win)
    profile = src.profile.copy()
    profile.update(
        height=data.shape[0], width=data.shape[1], transform=src.window_transform(win),
        compress="deflate", tiled=True, blockxsize=256, blockysize=256, driver="GTiff",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"[WorldPop] decupaj scris: {out_path.relative_to(C.ROOT)} {data.shape}")


def download_worldpop(force: bool = False):
    if C.WORLDPOP_CROP.exists() and not force:
        print(f"[WorldPop] OK ({C.WORLDPOP_CROP.relative_to(C.ROOT)} există)")
        return
    try:
        print("[WorldPop] citire pe fereastră prin /vsicurl/ ...")
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                   GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
        with rasterio.Env(**env), rasterio.open("/vsicurl/" + C.WORLDPOP_URL) as src:
            _write_crop(src, C.WORLDPOP_CROP)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[WorldPop] /vsicurl/ a eșuat ({e}); descarc fișierul complet ...")
    full = C.WORLDPOP_DIR / C.WORLDPOP_URL.rsplit("/", 1)[-1]
    full.parent.mkdir(parents=True, exist_ok=True)
    if not full.exists():
        tmp = full.with_suffix(".part")
        with urllib.request.urlopen(C.WORLDPOP_URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=16 * 1024 * 1024)
        tmp.rename(full)
    with rasterio.open(full) as src:
        _write_crop(src, C.WORLDPOP_CROP)


def main():
    ok = check_urbanev()
    download_worldpop(force="--force" in sys.argv)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
