"""Experimente de corectare și robustețe, evaluate DOAR prin CV pe train (setul de test nu e folosit).

Fiecare experiment pornește de la configurația aleasă de src.train (models/params.json) și schimbă un singur
lucru. Rezultatele se scriu în reports/experiments.json și sunt discutate în docs/experiments.md.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

from src import config as C
from src.baselines import B1
from src.split import split_table
from src.train import feature_list


def cv_eval(train, features, params, folds, target="y", weight=None, rounds=None):
    """MAE pe y (în puncte ocupate), oricare ar fi ținta modelului."""
    maes, iters, b1 = [], [], []
    for tr_idx, va_idx in folds:
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        if target == "occ":
            ytr, yva = tr["y"] / tr["n_piles"], va["y"] / va["n_piles"]
        else:
            ytr, yva = tr["y"], va["y"]
        w_tr = tr[weight] if weight else None
        w_va = va[weight] if weight else None
        dtr = lgb.Dataset(tr[features], ytr, weight=w_tr)
        dva = lgb.Dataset(va[features], yva, weight=w_va, reference=dtr)
        if rounds is None:
            booster = lgb.train(params, dtr, num_boost_round=C.MAX_BOOST_ROUNDS, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(C.EARLY_STOPPING_ROUNDS, verbose=False)])
            it = booster.best_iteration
        else:
            booster = lgb.train(params, dtr, num_boost_round=rounds)
            it = rounds
        pred = booster.predict(va[features], num_iteration=it)
        if target == "occ":
            pred = pred * va["n_piles"]
        maes.append(float(np.mean(np.abs(pred - va["y"]))))
        b1.append(float(np.mean(np.abs(B1().fit(tr).predict(va) - va["y"]))))
        iters.append(int(it))
    return {"cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
            "b1_cv_mae": float(np.mean(b1)), "gain_vs_b1": float(1 - np.mean(maes) / np.mean(b1)),
            "iters": float(np.mean(iters))}


def spatial_block_folds(train, stations, n_clusters=50):
    """Grupuri mari: KMeans pe coordonate (doar pentru grupare), apoi GroupKFold pe clustere."""
    st = stations.set_index("station_id").loc[train["station_id"].unique()]
    km = KMeans(n_clusters=n_clusters, random_state=C.SEED, n_init=10).fit(st[["latitude", "longitude"]])
    cl = pd.Series(km.labels_, index=st.index)
    groups = train["station_id"].map(cl)
    return list(GroupKFold(n_splits=C.N_FOLDS).split(train, groups=groups))


def main():
    table = pd.read_parquet(C.TRAIN_TABLE)
    train, _ = split_table(table)
    meta = json.loads(C.PARAMS_FILE.read_text())
    base_params = {k: v for k, v in meta["params"].items() if k != "monotone_constraints"}
    features = meta["features"]
    zone_folds = list(GroupKFold(n_splits=C.N_FOLDS).split(train, groups=train["zone_id"]))
    feats = pd.read_parquet(C.STATION_FEATURES_FILE)

    res = {}

    def run(name, desc, **kw):
        r = cv_eval(train, kw.pop("features", features), kw.pop("params", base_params),
                    kw.pop("folds", zone_folds), **kw)
        res[name] = {"description": desc, **r}
        print(f"{name:32s} CV MAE {r['cv_mae']:.4f} ± {r['cv_mae_std']:.4f}  (B1 {r['b1_cv_mae']:.4f}, "
              f"câștig {r['gain_vs_b1']:+.1%}, iter {r['iters']:.0f})")

    run("reference", "configurația aleasă de src.train")
    run("feature_set_base", "fără populație și POI", features=feature_list("base"))
    for f in C.CONTEXT_FEATURES:
        run(f"drop_{f}", f"fără {f}", features=[x for x in features if x != f])
    run("drop_competition", "fără feature-urile de competiție",
        features=[x for x in features if x not in C.COMPETITION_FEATURES])
    run("drop_power_kw", "fără power_kw", features=[x for x in features if x != "power_kw"])
    run("objective_l2", "obiectiv regression (L2) în loc de tweedie",
        params={**base_params, "objective": "regression"})
    run("objective_l1", "obiectiv regression_l1 (optimizează direct MAE)",
        params={**base_params, "objective": "regression_l1"})
    run("target_occupancy", "ținta = rata de ocupare y/n_piles (pondere n_piles), predicție × n_piles",
        target="occ", weight="n_piles", params={**base_params, "objective": "regression"})
    run("strong_regularization", "num_leaves 7, min_child_samples 300, lambda_l2 10",
        params={**base_params, "num_leaves": 7, "min_child_samples": 300, "lambda_l2": 10.0})
    run("lower_lr", "learning_rate 0.02", params={**base_params, "learning_rate": 0.02})
    mono = [C.MONOTONE.get(f, 0) for f in features]
    run("monotone_n_piles", "monotone_constraints: n_piles crescător",
        params={**base_params, "monotone_constraints": mono})

    # ce s-ar câștiga cu un feature suplimentar (nu e în model; doar sugestie)
    fs = feats.set_index("station_id")
    info = pd.read_csv(C.STATION_INFO).set_index("station_id")
    train_x = train.assign(fast_share=train["station_id"].map(info["fast_count"] / info["charge_count"]))
    r = cv_eval(train_x, features + ["fast_share"], base_params, zone_folds)
    res["extra_fast_share"] = {"description": "+ fast_share (ponderea punctelor DC); NU e în modelul final", **r}
    print(f"{'extra_fast_share':32s} CV MAE {r['cv_mae']:.4f} (câștig {r['gain_vs_b1']:+.1%})")

    # robustețe la autocorelarea spațială: blocuri spațiale mari în loc de zone TAZ
    block_folds = spatial_block_folds(train, fs.reset_index())
    run("spatial_blocks_kmeans50", "CV pe 50 de blocuri spațiale KMeans (în loc de zone TAZ)",
        folds=block_folds)

    C.REPORTS.mkdir(exist_ok=True)
    (C.REPORTS / "experiments.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
