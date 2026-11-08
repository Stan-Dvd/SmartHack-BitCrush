"""De ce contează atât de mult n_piles și power_kw? Trei experimente, doar pe stațiile din train (fără test).

A. Model doar de locație: același model de ocupare, fără n_piles și power_kw. Cât explică locul singur?
B. Infrastructură deduplicată: competiția calculată cu o singură înregistrare per stație fizică.
C. Experiment natural: stații diferite aflate la <= 100 m una de alta („micro-situri”). La locație practic
   fixă, cum variază ocuparea cu mărimea și puterea? Comparăm cu efectul dintre situri și cu ce prezice modelul.

Ieșiri: reports/size_experiments.json, reports/figures/size_within_site.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.neighbors import BallTree  # noqa: E402

from src import config as C  # noqa: E402
from src.build_dataset import build_station_features  # noqa: E402
from src.features_spatial import _rad  # noqa: E402
from src.load_urbanev import canonical_station_ids, load_station_info  # noqa: E402
from src.split import split_table  # noqa: E402
from src.train import baseline_cv, cv_folds, fit_booster, predict_y, run_cv  # noqa: E402

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"


def station_occupancy(df: pd.DataFrame, col: str) -> pd.Series:
    """Rata de ocupare zilnică pe stație: (5·medie zi lucrătoare + 2·medie weekend) / 7 / n_piles."""
    m = df.groupby(["station_id", "is_weekend"])[col].mean().unstack("is_weekend")
    n = df.groupby("station_id")["n_piles"].first()
    return ((5 * m[0] + 2 * m[1]) / 7 / n).rename(col)


def oof_predictions(train, features, params, mode, folds):
    """Predicții out-of-fold (puncte ocupate), cu numărul de runde ales prin early stopping pe fiecare fold."""
    oof = np.full(len(train), np.nan)
    for tr_idx, va_idx in folds:
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        b = fit_booster(tr, features, params, mode, C.MAX_BOOST_ROUNDS, valid=va)
        oof[va_idx] = predict_y(b, va, features, mode, num_iteration=b.best_iteration)
    return oof


def model_summary(train, oof, b1_oof):
    ok = ~np.isnan(oof)
    t = train[ok].assign(p=oof[ok], b1=b1_oof[ok])
    occ_real = station_occupancy(t, "y")
    occ_pred = station_occupancy(t, "p")
    full48 = t.groupby("station_id").size() == 48
    daily = lambda c: (t[t["station_id"].map(full48)].groupby(["station_id", "is_weekend"])[c].sum()  # noqa: E731
                       .unstack().pipe(lambda s: (5 * s[0] + 2 * s[1]) / 7))
    return {
        "cv_mae": float(np.mean(np.abs(t["p"] - t["y"]))),
        "spearman_occupancy_rate": float(spearmanr(occ_pred, occ_real).statistic),
        "spearman_daily_demand": float(spearmanr(daily("p"), daily("y")).statistic),
    }


def b1_oof_predictions(train, folds):
    from src.baselines import B1
    out = np.full(len(train), np.nan)
    for tr_idx, va_idx in folds:
        out[va_idx] = B1().fit(train.iloc[tr_idx]).predict(train.iloc[va_idx])
    return out


# ----------------------------------------------------------------------------- A + B
def experiments_ab(train, meta):
    folds = cv_folds(train)
    params = {k: v for k, v in meta["params"].items() if k != "monotone_constraints"}
    mode, feats_full = meta["target_mode"], meta["features"]
    mono = meta.get("monotone_dict", {})
    b1_oof = b1_oof_predictions(train, folds)
    b1_occ = station_occupancy(train.assign(b1=b1_oof), "b1")
    res = {"B1": {"cv_mae": float(np.mean(np.abs(b1_oof - train["y"]))),
                  "spearman_occupancy_rate": None,  # B1 dă aceeași ocupare tuturor stațiilor
                  "spearman_daily_demand": None},
           "note_B1": f"B1 prezice aceeași rată de ocupare pentru toate stațiile (std {b1_occ.std():.4f})"}
    # Spearman pe cererea zilnică pentru B1
    t = train.assign(b1=b1_oof)
    full48 = t.groupby("station_id").size() == 48
    s = t[t["station_id"].map(full48)].groupby(["station_id", "is_weekend"])[["y", "b1"]].sum().unstack()
    res["B1"]["spearman_daily_demand"] = float(spearmanr((5 * s["b1"][0] + 2 * s["b1"][1]),
                                                         (5 * s["y"][0] + 2 * s["y"][1])).statistic)

    variants = {
        "full": (feats_full, {**params, "monotone_constraints": [mono.get(f, 0) for f in feats_full]}),
        "location_only": ([f for f in feats_full if f not in C.STATION_FEATURES], params),
    }
    fold_maes = {}
    for name, (feats, p) in variants.items():
        oof = oof_predictions(train, feats, p, mode, folds)
        res[name] = {**model_summary(train, oof, b1_oof), "features": feats}
        fold_maes[name] = [float(np.mean(np.abs(oof[va] - train["y"].iloc[va]))) for _, va in folds]
        print(f"[A] {name:14s} CV MAE {res[name]['cv_mae']:.4f}  Spearman ocupare {res[name]['spearman_occupancy_rate']:.3f}"
              f"  Spearman cerere zilnică {res[name]['spearman_daily_demand']:.3f}")
    print(f"[A] B1             CV MAE {res['B1']['cv_mae']:.4f}  Spearman cerere zilnică "
          f"{res['B1']['spearman_daily_demand']:.3f}")

    # B: competiție pe infrastructură deduplicată (aceleași rânduri, aceleași fold-uri)
    feats_dedup = build_station_features(dedup=True, save=False).set_index("station_id")
    cols = [c for c in C.COMPETITION_FEATURES]
    train_d = train.copy()
    for c in cols:
        train_d[c] = train_d["station_id"].map(feats_dedup[c]).to_numpy()
    p_full = variants["full"][1]
    maes_d, _ = run_cv(train_d, feats_full, p_full, folds, mode)
    maes_r, _ = run_cv(train, feats_full, p_full, folds, mode)
    d = np.array(maes_d) - np.array(maes_r)
    res["dedup_infrastructure"] = {
        "cv_mae": float(np.mean(maes_d)), "cv_mae_reference": float(np.mean(maes_r)),
        "delta_vs_reference": float(d.mean()), "folds_better_than_reference": int((d < 0).sum()),
        "n_changed_stations": int((train.drop_duplicates("station_id").set_index("station_id")[cols]
                                   .ne(feats_dedup.loc[train["station_id"].unique(), cols]).any(axis=1)).sum()),
    }
    print(f"[B] dedup infrastructură: CV MAE {np.mean(maes_d):.4f} vs {np.mean(maes_r):.4f} "
          f"(Δ {d.mean():+.4f}, mai bun în {(d < 0).sum()}/5 fold-uri), stații cu feature-uri schimbate: "
          f"{res['dedup_infrastructure']['n_changed_stations']}")
    return res, folds, oof_predictions(train, feats_full, variants["full"][1], mode, folds)


# ----------------------------------------------------------------------------- C
def within_site(train, oof, radius_m=100):
    info = load_station_info().set_index("station_id")
    cov = pd.read_parquet(C.STATION_STATS_FILE).set_index("station_id")["coverage"]
    canon = canonical_station_ids(info.reset_index(), cov)
    t = train.assign(p=oof)
    st = pd.concat([station_occupancy(t, "y").rename("occ"), station_occupancy(t, "p").rename("occ_pred")], axis=1)
    st = st.join(info[["latitude", "longitude", "charge_count", "fast_count"]])
    st = st.join(train.drop_duplicates("station_id").set_index("station_id")[["power_kw"]])
    # o singură înregistrare per stație fizică (dublurile nu sunt stații diferite)
    st = st[st.index.isin(set(canon))]
    tree = BallTree(_rad(st["latitude"], st["longitude"]), metric="haversine")
    ind = tree.query_radius(_rad(st["latitude"], st["longitude"]), r=radius_m / C.EARTH_RADIUS_M)
    rows = [i for i, a in enumerate(ind) for _ in a]
    cols = [j for a in ind for j in a]
    from scipy.sparse import coo_matrix
    _, labels = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(st), len(st))))
    st["site"] = labels
    st["log2_piles"] = np.log2(st["charge_count"])
    st["log2_power"] = np.log2(st["power_kw"])
    multi = st[st.groupby("site")["occ"].transform("size") >= 2].copy()

    def fe_slopes(df, ycol, fixed_effects=True):
        X = df[["log2_piles", "log2_power"]].copy()
        y = df[ycol].copy()
        if fixed_effects:
            X = X - X.groupby(df["site"]).transform("mean")
            y = y - y.groupby(df["site"]).transform("mean")
        else:
            X = X - X.mean()
            y = y - y.mean()
        beta, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
        return beta

    rng = np.random.default_rng(C.SEED)
    sites = multi["site"].unique()
    boot = []
    for _ in range(2000):
        pick = rng.choice(sites, size=len(sites), replace=True)
        b = pd.concat([multi[multi["site"] == s].assign(site=k) for k, s in enumerate(pick)])
        if b["log2_piles"].groupby(b["site"]).std().fillna(0).sum() == 0:
            continue
        boot.append(fe_slopes(b, "occ"))
    boot = np.array(boot)
    ci = lambda k: [float(np.quantile(boot[:, k], 0.025)), float(np.quantile(boot[:, k], 0.975))]  # noqa: E731

    within_real = fe_slopes(multi, "occ")
    within_pred = fe_slopes(multi, "occ_pred")
    between_real = fe_slopes(st, "occ", fixed_effects=False)
    between_pred = fe_slopes(st, "occ_pred", fixed_effects=False)
    # pe perechi: când stația mai mare are ocupare mai mare?
    pairs = []
    for _, g in multi.groupby("site"):
        g = g.sort_values("charge_count")
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                if b["charge_count"] > a["charge_count"]:
                    pairs.append((b["log2_piles"] - a["log2_piles"], b["occ"] - a["occ"], b["occ_pred"] - a["occ_pred"]))
    pairs = np.array(pairs)
    res = {
        "radius_m": radius_m,
        "n_stations_in_multi_station_sites": int(len(multi)),
        "n_sites": int(len(sites)),
        "n_pairs_bigger_vs_smaller": int(len(pairs)),
        "share_pairs_bigger_has_higher_occupancy_real": float(np.mean(pairs[:, 1] > 0)) if len(pairs) else None,
        "share_pairs_bigger_has_higher_occupancy_model": float(np.mean(pairs[:, 2] > 0)) if len(pairs) else None,
        "slope_occ_per_doubling_piles": {
            "within_site_real": float(within_real[0]), "within_site_real_ci95": ci(0),
            "within_site_model": float(within_pred[0]),
            "between_sites_real": float(between_real[0]), "between_sites_model": float(between_pred[0]),
        },
        "slope_occ_per_doubling_power": {
            "within_site_real": float(within_real[1]), "within_site_real_ci95": ci(1),
            "within_site_model": float(within_pred[1]),
            "between_sites_real": float(between_real[1]), "between_sites_model": float(between_pred[1]),
        },
        "mean_occupancy": float(st["occ"].mean()),
    }
    print(f"[C] micro-situri (<= {radius_m} m): {len(sites)} situri, {len(multi)} stații, {len(pairs)} perechi")
    for k in ("slope_occ_per_doubling_piles", "slope_occ_per_doubling_power"):
        v = res[k]
        print(f"[C] {k}: în sit real {v['within_site_real']:+.3f} (IC95 {v['within_site_real_ci95'][0]:+.3f}…"
              f"{v['within_site_real_ci95'][1]:+.3f}), în sit model {v['within_site_model']:+.3f}; "
              f"între situri real {v['between_sites_real']:+.3f}, model {v['between_sites_model']:+.3f}")
    print(f"[C] perechi în care stația mai mare e mai plină: real {res['share_pairs_bigger_has_higher_occupancy_real']:.0%}, "
          f"model {res['share_pairs_bigger_has_higher_occupancy_model']:.0%}")

    if len(pairs):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, k, title in ((axes[0], 1, "real"), (axes[1], 2, "prezis de model")):
            ax.scatter(pairs[:, 0], pairs[:, k], s=18, color=BLUE, alpha=0.7, edgecolor="#fcfcfb", linewidth=0.5)
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.set_title(f"Diferența de ocupare în același micro-sit ({title})", loc="left", fontsize=10)
            ax.set_xlabel("log2(puncte stația mai mare / puncte stația mai mică)")
            ax.grid(color=GRID, linewidth=0.6)
        axes[0].set_ylabel("ocupare mai mare − ocupare mai mică")
        fig.tight_layout()
        fig.savefig(C.FIGURES / "size_within_site.png", bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(fig)
    return res


def main():
    table = pd.read_parquet(C.TRAIN_TABLE)
    train, _ = split_table(table)
    meta = json.loads(C.PARAMS_FILE.read_text())
    res_ab, folds, oof_full = experiments_ab(train, meta)
    res_c = within_site(train, oof_full)
    out = {"A_B": res_ab, "C_within_site": res_c}
    (C.REPORTS / "size_experiments.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
