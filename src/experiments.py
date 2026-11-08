"""Experimente de corectare și robustețe, evaluate DOAR prin CV pe train (setul de test nu e folosit).

Fiecare experiment pornește de la configurația aleasă de src.train (models/params.json) și schimbă un singur
lucru. Comparațiile sunt pe aceleași fold-uri (perechi): raportăm diferența medie față de referință și
în câte din cele 5 fold-uri experimentul e mai bun. Rezultatele: reports/experiments.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from src import config as C
from src.split import split_table
from src.train import baseline_cv, feature_list, run_cv


def spatial_block_folds(train, stations, n_clusters=50):
    """Grupuri mari: KMeans pe coordonate (doar pentru grupare, nu ca feature), apoi GroupKFold pe clustere."""
    st = stations.set_index("station_id").loc[train["station_id"].unique()]
    km = KMeans(n_clusters=n_clusters, random_state=C.SEED, n_init=10).fit(st[["latitude", "longitude"]])
    groups = train["station_id"].map(pd.Series(km.labels_, index=st.index))
    return list(GroupKFold(n_splits=C.N_FOLDS).split(train, groups=groups))


def main():
    table = pd.read_parquet(C.TRAIN_TABLE)
    train, _ = split_table(table)
    meta = json.loads(C.PARAMS_FILE.read_text())
    ref_params = {k: v for k, v in meta["params"].items() if k != "monotone_constraints"}
    ref_mono = meta.get("monotone_dict", {})
    ref_mode = meta.get("target_mode", "y")
    features = meta["features"]
    zone_folds = list(GroupKFold(n_splits=C.N_FOLDS).split(train, groups=train["zone_id"]))
    stations = pd.read_parquet(C.STATION_FEATURES_FILE)
    b1 = baseline_cv(train, zone_folds)

    res, ref_folds = {}, None

    def run(name, desc, features=features, params=ref_params, mode=ref_mode, folds=zone_folds, data=train,
            b1_folds=b1, mono=ref_mono):
        nonlocal ref_folds
        if mono:  # vectorul de constrângeri se reconstruiește pentru lista de feature-uri a experimentului
            params = {**params, "monotone_constraints": [mono.get(f, 0) for f in features]}
        maes, iters = run_cv(data, features, params, folds, mode)
        r = {"description": desc, "cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
             "fold_maes": maes, "b1_cv_mae": float(np.mean(b1_folds)),
             "gain_vs_b1": float(1 - np.mean(maes) / np.mean(b1_folds)), "iters": float(np.mean(iters))}
        if ref_folds is None:
            ref_folds = maes
        elif folds is zone_folds:
            d = np.array(maes) - np.array(ref_folds)
            r["delta_vs_reference"] = float(d.mean())
            r["folds_better_than_reference"] = int((d < 0).sum())
        res[name] = r
        extra = (f", Δ ref {r['delta_vs_reference']:+.4f}, mai bun în {r['folds_better_than_reference']}/5 fold-uri"
                 if "delta_vs_reference" in r else "")
        print(f"{name:28s} CV MAE {r['cv_mae']:.4f} ± {r['cv_mae_std']:.4f} (câștig vs B1 {r['gain_vs_b1']:+.1%}"
              f"{extra})")

    run("reference", f"configurația aleasă de src.train ({ref_mode}, {ref_params['objective']})")

    # --- ablații pe feature-uri
    run("feature_set_base", "fără populație și POI", features=feature_list("base"))
    run("drop_poi_all", "fără cele 3 feature-uri POI", features=[f for f in features if f not in C.POI_FEATURES])
    run("drop_pop_1km", "fără pop_1km", features=[f for f in features if f != "pop_1km"])
    run("drop_competition", "fără feature-urile de competiție",
        features=[f for f in features if f not in C.COMPETITION_FEATURES])
    run("drop_power_kw", "fără power_kw", features=[f for f in features if f != "power_kw"])

    # --- ținta și obiectivul
    run("target_y_tweedie", "ținta y directă, tweedie 1.2 (abordarea inițială din plan)", mode="y",
        params={**ref_params, "objective": "tweedie", "tweedie_variance_power": 1.2})
    run("target_y_l2", "ținta y directă, L2", mode="y", params={**ref_params, "objective": "regression"})
    if ref_mode == "occ":
        run("occ_l2", "ocupare, L2", params={**ref_params, "objective": "regression"})
        run("occ_cross_entropy", "ocupare, cross-entropy", params={**ref_params, "objective": "cross_entropy"})

    # --- regularizare
    run("lower_lr", "learning_rate 0.02", params={**ref_params, "learning_rate": 0.02})
    run("lambda_l2_10", "lambda_l2 = 10", params={**ref_params, "lambda_l2": 10.0})

    # --- constrângeri monotone (pe ieșirea modelului = rata de ocupare în modul "occ")
    run("no_monotone", "fără constrângeri monotone", mono={})
    run("diminishing_returns_n_piles", "ocupare necrescătoare în n_piles (randament descrescător per punct) "
        "+ ocupare nedescrescătoare în power_kw", mono={"n_piles": -1, "power_kw": 1})

    # --- un feature suplimentar (NU e în model; doar sugestie pentru faza următoare)
    info = pd.read_csv(C.STATION_INFO).set_index("station_id")
    train_x = train.assign(fast_share=train["station_id"].map(info["fast_count"] / info["charge_count"]))
    run("extra_fast_share", "+ fast_share (ponderea punctelor DC); NU e în modelul final",
        features=features + ["fast_share"], data=train_x)

    # --- robustețe la autocorelarea spațială: blocuri spațiale mari în loc de zone TAZ
    block_folds = spatial_block_folds(train, stations)
    run("spatial_blocks_kmeans50", "CV pe 50 de blocuri spațiale KMeans (în loc de zone TAZ)",
        folds=block_folds, b1_folds=baseline_cv(train, block_folds))

    C.REPORTS.mkdir(exist_ok=True)
    (C.REPORTS / "experiments.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
