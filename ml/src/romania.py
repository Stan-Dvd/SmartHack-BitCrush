"""Setul românesc (data/raw/romania/romania_ev_5min.csv): randomizarea lui `busy` și antrenarea pipeline-ului.

ATENȚIE: `busy` din fișierul primit e 100% sintetic (busy_is_synthetic = 1). Comanda `randomize` îl înlocuiește
cu valori aleatoare, fără nicio legătură cu feature-urile. Antrenarea pe aceste date e un CONTROL NEGATIV:
un pipeline corect nu are voie să găsească semnal de locație acolo unde nu există. Rezultatele NU descriu cererea
reală din România și nu trebuie prezentate ca atare.

    python -m src.romania randomize   # -> data/raw/romania/romania_ev_5min_random.csv (originalul rămâne)
    python -m src.romania train       # -> models/romania/, reports/romania/ (zone = județe)
    python -m src.romania train --csv data/raw/romania/alt_fisier.csv
    python -m src.romania diagnose    # grafice: overfitting/underfitting + predicții vs realitate (reports/romania/)
"""
import argparse
import itertools
import json
import zlib

import numpy as np
import pandas as pd

from src import config as C
from src.baselines import B0, B1
from src.evaluate import compute_metrics, daily_hours, table_md
from src.split import make_split, split_table
from src.train import (
    GRID,
    OBJECTIVES,
    baseline_cv,
    cv_folds,
    fit_booster,
    make_params,
    predict_y,
    run_cv,
)

RO_DIR = C.RAW / "romania"
RO_CSV = RO_DIR / "romania_ev_5min.csv"
RO_RANDOM_CSV = RO_DIR / "romania_ev_5min_random.csv"
RO_PROCESSED = C.PROCESSED / "romania"
RO_MODELS = C.MODELS / "romania"
RO_REPORTS = C.REPORTS / "romania"
STATIC = [f for f in C.FEATURES if f not in C.TEMPORAL_FEATURES]
CHUNK = 2_000_000


# ----------------------------------------------------------------------------- randomizare
def _station_params(station_id: str, seed: int):
    """Ocupare medie și profil orar aleatoare pentru o stație (deterministe după ID și seed)."""
    rng = np.random.default_rng([seed, zlib.crc32(station_id.encode())])
    base = rng.beta(2, 8)  # ocupare medie a stației (medie 0,2), FĂRĂ legătură cu feature-urile
    prof = rng.lognormal(0.0, 0.35, size=(2, 24))  # multiplicator pe (is_weekend, oră)
    return base, prof / prof.mean()


def randomize(src=RO_CSV, dst=RO_RANDOM_CSV, seed=C.SEED):
    params = {}
    first = True
    n = 0
    for k, ch in enumerate(pd.read_csv(src, chunksize=CHUNK, dtype={"station_id": str, "zone_id": str})):
        for sid in ch["station_id"].unique():
            if sid not in params:
                params[sid] = _station_params(sid, seed)
        base = ch["station_id"].map({s: v[0] for s, v in params.items()}).to_numpy()
        prof = np.stack([params[s][1][w, h] for s, w, h in
                         zip(ch["station_id"], ch["is_weekend"].to_numpy(), ch["hour"].to_numpy())])
        p = np.clip(base * prof, 0.0, 0.98)
        rng = np.random.default_rng([seed, k])
        ch["busy"] = rng.binomial(ch["n_piles"].to_numpy().astype(int), p)
        ch.to_csv(dst, mode="w" if first else "a", header=first, index=False)
        first = False
        n += len(ch)
        print(f"  {n:,} rânduri scrise")
    print(f"scris: {dst.relative_to(C.ROOT)} (busy aleator, seed {seed}); originalul a rămas neschimbat")


# ----------------------------------------------------------------------------- tabelul de antrenare
def build_table(csv=RO_RANDOM_CSV) -> pd.DataFrame:
    parts, static = [], []
    for ch in pd.read_csv(csv, chunksize=CHUNK, dtype={"station_id": str, "zone_id": str}):
        ch = ch[ch["busy"].notna()]
        g = ch.groupby(["station_id", "hour", "is_weekend"])["busy"].agg(["sum", "count"])
        parts.append(g)
        static.append(ch.groupby("station_id")[["zone_id", *STATIC]].first())
    cells = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
    cells = cells[cells["count"] >= C.MIN_CELL_OBS]
    table = (cells.assign(y=cells["sum"] / cells["count"], n_obs=cells["count"]).drop(columns=["sum", "count"])
             .reset_index())
    st = pd.concat(static).groupby(level=0).first()
    table = table.merge(st, left_on="station_id", right_index=True, how="left", validate="many_to_one")
    for c in STATIC:
        table[c] = pd.to_numeric(table[c])
    assert table[[*C.FEATURES, "y"]].notna().all().all()
    RO_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_parquet(RO_PROCESSED / "train_table.parquet", index=False)
    print(f"tabel: {len(table):,} rânduri, {table.station_id.nunique()} stații, {table.zone_id.nunique()} județe")
    return table


# ----------------------------------------------------------------------------- antrenare și evaluare
def train_and_evaluate(table: pd.DataFrame):
    split = make_split(table)  # 20% din județe în test, seed 42
    (RO_PROCESSED / "split.json").write_text(json.dumps(split, indent=1, ensure_ascii=False))
    train, test = split_table(table, split)
    print(f"split: {len(split['train_zones'])} județe train / {len(split['test_zones'])} test "
          f"({', '.join(map(str, split['test_zones']))}); stații {train.station_id.nunique()} / "
          f"{test.station_id.nunique()}")

    feats = list(C.FEATURES)
    folds = cv_folds(train)
    b1_cv = baseline_cv(train, folds)
    rows = []
    for (mode, obj), extra in itertools.product(OBJECTIVES, [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]):
        params = make_params(obj, extra, feats, C.USE_MONOTONE)
        maes, iters = run_cv(train, feats, params, folds, mode)
        rows.append({"target_mode": mode, **obj, **extra, "cv_mae": np.mean(maes), "best_iter_mean": np.mean(iters),
                     "fold_maes": maes})
    cv = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    best = cv.iloc[0]
    mode = best["target_mode"]
    obj = {"objective": best["objective"]}
    if best["objective"] == "tweedie":
        obj["tweedie_variance_power"] = float(best["tweedie_variance_power"])
    extra = {k: int(best[k]) for k in GRID}
    params = make_params(obj, extra, feats, C.USE_MONOTONE)
    n_rounds = max(1, int(round(best["best_iter_mean"])))
    _, _, tr_maes = run_cv(train, feats, params, folds, mode, return_train_mae=True)
    booster = fit_booster(train, feats, params, mode, n_rounds)

    pred = test.copy()
    pred["pred_B0"] = B0().fit(train).predict(test)
    pred["pred_B1"] = B1().fit(train).predict(test)
    pred["pred_LGBM"] = predict_y(booster, test, feats, mode)
    metrics = compute_metrics(pred)
    info = {
        "n_test_rows": int(len(test)), "n_test_stations": int(test["station_id"].nunique()),
        "n_test_zones": int(test["zone_id"].nunique()), "n_test_stations_full": int(len(daily_hours(pred, "y"))),
        "y_mean_test": float(test["y"].mean()), "daily_mean_test": float(daily_hours(pred, "y").mean()),
    }
    meta = {"WARNING": "busy aleator (control negativ); NU descrie cererea reală din România",
            "target_mode": mode, "params": params, "num_boost_round": n_rounds,
            "cv_mae": float(best["cv_mae"]), "b1_cv_mae": float(np.mean(b1_cv)),
            "cv_train_fold_mae": float(np.mean(tr_maes)), "overfit_gap": float(best["cv_mae"] - np.mean(tr_maes))}
    RO_MODELS.mkdir(parents=True, exist_ok=True)
    RO_REPORTS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(RO_MODELS / "lgbm.txt"))
    (RO_MODELS / "params.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (RO_REPORTS / "metrics.json").write_text(json.dumps({"metrics": metrics, "info": info, "model": meta},
                                                         indent=2, ensure_ascii=False))
    cv.to_csv(RO_REPORTS / "cv_results.csv", index=False)
    print(f"\nCV: LightGBM {best['cv_mae']:.4f} vs B1 {np.mean(b1_cv):.4f} "
          f"(câștig {1 - best['cv_mae'] / np.mean(b1_cv):+.1%}); memorare {meta['overfit_gap']:.3f}; "
          f"configurație {mode} {json.dumps({**obj, **extra})}, {n_rounds} runde")
    print(table_md(metrics, info))
    return metrics, meta


# ----------------------------------------------------------------------------- diagnostic și grafice
def noise_floor_csv(csv, station_ids) -> float:
    """Plafonul de zgomot: MAE între mediile pe săptămâni pare și impare per celulă (stație, oră, tip zi), / 2."""
    ids = set(station_ids)
    parts = []
    for ch in pd.read_csv(csv, chunksize=CHUNK, usecols=["timestamp", "station_id", "busy", "hour", "is_weekend"],
                          dtype={"station_id": str}):
        ch = ch[ch["station_id"].isin(ids) & ch["busy"].notna()]
        half = pd.to_datetime(ch["timestamp"]).dt.isocalendar().week.to_numpy() % 2
        parts.append(ch.assign(half=half).groupby(["station_id", "hour", "is_weekend", "half"])["busy"]
                     .agg(["sum", "count"]))
    g = pd.concat(parts).groupby(level=[0, 1, 2, 3]).sum()
    m = (g["sum"] / g["count"]).unstack("half").dropna()
    return float(np.abs(m[0] - m[1]).mean() / 2)


def diagnose(csv=RO_RANDOM_CSV):
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    from src import diagnostics as D

    table_path = RO_PROCESSED / "train_table.parquet"
    table = pd.read_parquet(table_path) if table_path.exists() else build_table(csv)
    split = json.loads((RO_PROCESSED / "split.json").read_text())
    meta = json.loads((RO_MODELS / "params.json").read_text())
    train, test = split_table(table, split)
    feats, mode, n_rounds = list(C.FEATURES), meta["target_mode"], meta["num_boost_round"]
    params = meta["params"]
    unconstrained = {k: v for k, v in params.items() if k != "monotone_constraints"}
    var = {"model final (monoton)": (params, n_rounds), "fără constrângeri (comparație)": (unconstrained, n_rounds)}
    floor = noise_floor_csv(csv, train["station_id"].unique())
    out, curves, lc = D.compute_diagnostics(train, test, feats, mode, var, "model final (monoton)", floor)
    fig_dir = RO_REPORTS / "figures"
    D.plot_diagnostics(
        out, curves, lc, styles={"model final (monoton)": D.ORANGE, "fără constrângeri (comparație)": D.GRAY},
        final_name="modelul final",
        bar_labels={"B0": "B0\nprofil oraș", "B1": "B1\nocupare × puncte",
                    "fără constrângeri (comparație)": "fără\nconstrângeri", "model final (monoton)": "model final"},
        path=fig_dir / "fit_diagnostics.png",
        title="România (busy ALEATOR, control negativ): overfitting și underfitting",
    )
    (RO_REPORTS / "fit_diagnostics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # --- predicții vs realitate pe stațiile din test
    booster = fit_booster(train, feats, params, mode, n_rounds)
    pred = test.assign(p_lgbm=predict_y(booster, test, feats, mode), p_b1=B1().fit(train).predict(test))
    real_d, lgbm_d, b1_d = (daily_hours(pred, c) for c in ("y", "p_lgbm", "p_b1"))
    piles = pred.groupby("station_id")["n_piles"].first().loc[real_d.index]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    lim = 1.05 * max(real_d.max(), lgbm_d.max(), b1_d.max())
    ax.plot([0, lim], [0, lim], color=D.INK2, linewidth=0.8, linestyle=":", label="predicție perfectă")
    ax.scatter(real_d, b1_d, s=30, color=D.AQUA, edgecolor=D.SURF, linewidth=1.5,
               label=f"B1 (Spearman {spearmanr(real_d, b1_d).statistic:.2f})")
    ax.scatter(real_d, lgbm_d, s=30, color=D.ORANGE, edgecolor=D.SURF, linewidth=1.5,
               label=f"LightGBM (Spearman {spearmanr(real_d, lgbm_d).statistic:.2f})")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("ore de încărcare pe zi, reale")
    ax.set_ylabel("ore de încărcare pe zi, prezise")
    ax.set_title("D. Cererea zilnică pe stație (test)", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    occ_r, occ_p = real_d / 24 / piles, lgbm_d / 24 / piles
    order = np.argsort(occ_r.to_numpy())
    xs = np.arange(len(order))
    ax.scatter(xs, occ_r.to_numpy()[order], s=26, color=D.BLUE, edgecolor=D.SURF, linewidth=1.2, label="reală")
    ax.scatter(xs, occ_p.to_numpy()[order], s=26, color=D.ORANGE, edgecolor=D.SURF, linewidth=1.2,
               label="prezisă (LightGBM)")
    ax.set_xlabel("stațiile din test, ordonate după ocuparea reală")
    ax.set_ylabel("rata de ocupare")
    ax.set_title("E. Ocuparea pe stație: reală vs prezisă", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    wd = pred[pred["is_weekend"] == 0]
    prof = (wd.groupby("hour")[["y", "p_lgbm", "p_b1"]].sum().div(wd.groupby("hour")["n_piles"].sum(), axis=0))
    ax.plot(prof.index, prof["y"], color=D.BLUE, label="real")
    ax.plot(prof.index, prof["p_lgbm"], color=D.ORANGE, label="LightGBM")
    ax.plot(prof.index, prof["p_b1"], color=D.AQUA, linestyle="--", label="B1")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(0, 1.3 * prof.max().max())
    ax.set_xlabel("ora (zi lucrătoare)")
    ax.set_ylabel("rata de ocupare medie")
    ax.set_title("F. Profilul orar mediu (test)", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("România (busy ALEATOR, control negativ): predicții vs realitate pe cele "
                 f"{len(real_d)} stații din test", x=0.01, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "predictions_vs_real.png", bbox_inches="tight")
    plt.close(fig)
    print(f"figuri: {(fig_dir / 'fit_diagnostics.png').relative_to(C.ROOT)}, "
          f"{(fig_dir / 'predictions_vs_real.png').relative_to(C.ROOT)}")
    print(f"ocupare pe stație (test): reală între {occ_r.min():.3f} și {occ_r.max():.3f}; "
          f"prezisă între {occ_p.min():.3f} și {occ_p.max():.3f}; Spearman {spearmanr(occ_r, occ_p).statistic:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["randomize", "train", "diagnose"])
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    csv = C.ROOT / a.csv if a.csv else RO_RANDOM_CSV
    if a.command == "randomize":
        randomize()
    elif a.command == "train":
        train_and_evaluate(build_table(csv))
    else:
        diagnose(csv)


if __name__ == "__main__":
    main()
