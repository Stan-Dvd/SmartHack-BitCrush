"""Diagnostic de potrivire (overfitting / underfitting) pentru prezentare.

A. Loss pe runde de boosting: MAE pe fold-urile de antrenare vs validare (CV pe zone), modelul final
   (monoton) față de modelul anterior (monoton doar pe power_kw).
B. Curba de învățare: MAE în funcție de numărul de stații de antrenare.
C. MAE pe antrenare / CV / test pentru B0, B1, modelul anterior și modelul final, plus plafonul de zgomot:
   eroarea unui model care ar cunoaște perfect fiecare stație (estimată din săptămâni pare vs impare).

Ieșiri: reports/figures/fit_diagnostics.png, reports/fit_diagnostics.json
"""
import json
from multiprocessing import Pool

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.baselines import B0, B1  # noqa: E402
from src.load_urbanev import read_station_series  # noqa: E402
from src.split import split_table  # noqa: E402
from src.train import cv_folds, fit_booster, label_weight, load_model, predict_y  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 110,
})
ROUNDS = 800  # peste optimul ales, ca să se vadă ce se întâmplă dacă antrenăm prea mult


def variants(meta):
    base = {k: v for k, v in meta["params"].items() if k != "monotone_constraints"}
    feats = meta["features"]
    final = {**base, "monotone_constraints": [C.MONOTONE.get(f, 0) for f in feats]}
    # modelul anterior: configurația aleasă înainte de constrângerile pe toate feature-urile
    old = {**base, "num_leaves": 31, "min_child_samples": 1000,
           "monotone_constraints": [1 if f == "power_kw" else 0 for f in feats]}
    return {"model final (monoton)": (final, meta["num_boost_round"]),
            "model anterior (monoton doar pe putere)": (old, 108)}


def loss_curves(train, feats, mode, params, folds):
    """MAE pe y (puncte ocupate) la fiecare rundă, pe fold-urile de antrenare și de validare."""
    tr_curves, va_curves = [], []
    for a, b in folds:
        tr, va = train.iloc[a], train.iloc[b]
        lab, w = label_weight(tr, mode)
        vlab, vw = label_weight(va, mode)
        dtr = lgb.Dataset(tr[feats], lab, weight=w)
        dva = lgb.Dataset(va[feats], vlab, weight=vw, reference=dtr)
        hist = {}
        lgb.train({**params, "metric": "l1"}, dtr, num_boost_round=ROUNDS, valid_sets=[dtr, dva],
                  valid_names=["train", "valid"], callbacks=[lgb.record_evaluation(hist)])
        # L1 ponderat cu n_piles pe ocupare = Σ|eroare y| / Σ n_piles  ->  MAE pe y = L1 · Σn / N
        k_tr = tr["n_piles"].sum() / len(tr) if mode == "occ" else 1.0
        k_va = va["n_piles"].sum() / len(va) if mode == "occ" else 1.0
        tr_curves.append(np.array(hist["train"]["l1"]) * k_tr)
        va_curves.append(np.array(hist["valid"]["l1"]) * k_va)
    return np.mean(tr_curves, axis=0), np.mean(va_curves, axis=0)


def learning_curve(train, feats, mode, params, folds, fracs=(0.25, 0.5, 0.75, 1.0), repeats=5):
    """Pentru fiecare fracțiune: `repeats` eșantioane aleatoare de zone de antrenare (în fiecare fold).

    Raportăm media pe repetări și intervalul min–max, ca să se vadă cât din formă e zgomot de eșantionare.
    La 100% toate repetările sunt identice (nu mai e nimic de eșantionat).
    """
    rng = np.random.default_rng(C.SEED)
    out = []
    for frac in fracs:
        rep_tr, rep_va, n_st = [], [], []
        for _ in range(repeats if frac < 1 else 1):
            trm, vam = [], []
            for a, b in folds:
                tr_all, va = train.iloc[a], train.iloc[b]
                zones = tr_all["zone_id"].unique()
                keep = rng.choice(zones, size=max(1, int(round(frac * len(zones)))), replace=False)
                tr = tr_all[tr_all["zone_id"].isin(keep)]
                bst = fit_booster(tr, feats, params, mode, C.MAX_BOOST_ROUNDS, valid=va)
                it = bst.best_iteration
                trm.append(np.abs(predict_y(bst, tr, feats, mode, it) - tr["y"]).mean())
                vam.append(np.abs(predict_y(bst, va, feats, mode, it) - va["y"]).mean())
                n_st.append(tr["station_id"].nunique())
            rep_tr.append(np.mean(trm))
            rep_va.append(np.mean(vam))
        out.append({"frac": frac, "n_stations": float(np.mean(n_st)),
                    "train_mae": float(np.mean(rep_tr)), "train_min": float(np.min(rep_tr)),
                    "train_max": float(np.max(rep_tr)),
                    "cv_mae": float(np.mean(rep_va)), "cv_min": float(np.min(rep_va)), "cv_max": float(np.max(rep_va)),
                    "repeats": len(rep_va)})
    return out


def _halves(station_id):
    s = read_station_series(station_id).dropna()
    s = s.assign(hour=s["time"].dt.hour, is_weekend=(s["time"].dt.dayofweek >= 5).astype(int),
                 half=s["time"].dt.isocalendar().week % 2)
    g = s.groupby(["hour", "is_weekend", "half"])["busy"].mean().unstack("half")
    g = g.dropna()
    return station_id, np.abs(g[0] - g[1]).sum(), len(g)


def noise_floor(station_ids):
    """MAE între media pe săptămâni pare și media pe săptămâni impare, per celulă (stație, oră, tip zi).

    Fiecare jumătate are de 2 ori mai puține date decât media completă; diferența a două jumătăți are o
    abatere de ~2 ori mai mare decât eroarea mediei complete față de valoarea adevărată, deci
    plafon ≈ MAE(jumătăți) / 2.
    """
    with Pool(C.N_JOBS) as pool:
        res = pool.map(_halves, station_ids, chunksize=8)
    total = sum(r[1] for r in res)
    n = sum(r[2] for r in res)
    return float(total / n / 2)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    table = pd.read_parquet(C.TRAIN_TABLE)
    train, test = split_table(table)
    _, meta = load_model()
    feats, mode = meta["features"], meta["target_mode"]
    folds = cv_folds(train)
    var = variants(meta)
    out = {"loss_curves": {}, "bars": {}}

    # A. loss pe runde
    curves = {}
    for name, (params, n_rounds) in var.items():
        tr_c, va_c = loss_curves(train, feats, mode, params, folds)
        curves[name] = (tr_c, va_c, n_rounds)
        out["loss_curves"][name] = {"chosen_rounds": n_rounds, "train_mae_at_chosen": float(tr_c[n_rounds - 1]),
                                    "cv_mae_at_chosen": float(va_c[n_rounds - 1]),
                                    "cv_mae_min": float(va_c.min()), "round_of_min": int(va_c.argmin() + 1),
                                    "train_mae_at_end": float(tr_c[-1]), "cv_mae_at_end": float(va_c[-1])}
        print(f"[A] {name}: la runda {n_rounds}: antrenare {tr_c[n_rounds-1]:.3f}, validare {va_c[n_rounds-1]:.3f}; "
              f"la runda {ROUNDS}: antrenare {tr_c[-1]:.3f}, validare {va_c[-1]:.3f}")

    # B. curba de învățare (modelul final)
    final_params = var["model final (monoton)"][0]
    lc = learning_curve(train, feats, mode, final_params, folds)
    out["learning_curve"] = lc
    for r in lc:
        print(f"[B] {r['n_stations']:.0f} stații: antrenare {r['train_mae']:.3f} ({r['train_min']:.3f}–{r['train_max']:.3f}), "
              f"validare {r['cv_mae']:.3f} ({r['cv_min']:.3f}–{r['cv_max']:.3f})")

    # C. antrenare / CV / test + plafon de zgomot
    floor = noise_floor(train["station_id"].unique().tolist())
    out["noise_floor_mae"] = floor
    print(f"[C] plafon de zgomot (stație cunoscută perfect): MAE ≈ {floor:.3f}")
    for name, model in (("B0", B0), ("B1", B1)):
        cv = np.mean([mae(model().fit(train.iloc[a]).predict(train.iloc[b]), train["y"].iloc[b]) for a, b in folds])
        tr_m = np.mean([mae(model().fit(train.iloc[a]).predict(train.iloc[a]), train["y"].iloc[a]) for a, b in folds])
        te = mae(model().fit(train).predict(test), test["y"])
        out["bars"][name] = {"train": float(tr_m), "cv": float(cv), "test": te}
    for name, (params, n_rounds) in var.items():
        tr_c, va_c, _ = curves[name]
        bst = fit_booster(train, feats, params, mode, n_rounds)
        te = mae(predict_y(bst, test, feats, mode), test["y"])
        out["bars"][name] = {"train": float(tr_c[n_rounds - 1]), "cv": float(va_c[n_rounds - 1]), "test": te}
    for k, v in out["bars"].items():
        print(f"[C] {k:42s} antrenare {v['train']:.3f}  CV {v['cv']:.3f}  test {v['test']:.3f}")

    # ---------------------------------------------------------------- figura
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    styles = {"model final (monoton)": ORANGE, "model anterior (monoton doar pe putere)": GRAY}
    x = np.arange(1, ROUNDS + 1)
    for name, (tr_c, va_c, n_rounds) in curves.items():
        c = styles[name]
        ax.plot(x, va_c, color=c, label=f"{name}: validare")
        ax.plot(x, tr_c, color=c, linestyle="--", linewidth=1.5, label=f"{name}: antrenare")
        ax.axvline(n_rounds, color=c, linewidth=0.8, linestyle=":")
    b1_cv = out["bars"]["B1"]["cv"]
    ax.axhline(b1_cv, color=AQUA, linewidth=1.2, label=f"B1 (validare) = {b1_cv:.2f}")
    ax.set_xlabel("runde de boosting (linia punctată = numărul ales)")
    ax.set_ylabel("MAE (puncte ocupate)")
    ax.set_title("A. Loss în timpul antrenării (CV pe zone)", loc="left")
    ax.set_ylim(0.3, 2.4)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ns = [r["n_stations"] for r in lc]
    ax.fill_between(ns, [r["cv_min"] for r in lc], [r["cv_max"] for r in lc], color=ORANGE, alpha=0.18, linewidth=0)
    ax.fill_between(ns, [r["train_min"] for r in lc], [r["train_max"] for r in lc], color=ORANGE, alpha=0.10,
                    linewidth=0, label="variație între 5 eșantioane (min–max)")
    ax.plot(ns, [r["cv_mae"] for r in lc], color=ORANGE, marker="o", markersize=7, label="validare (zone nevăzute)")
    ax.plot(ns, [r["train_mae"] for r in lc], color=ORANGE, linestyle="--", marker="o", markersize=6,
            markerfacecolor=SURF, label="antrenare")
    ax.axhline(b1_cv, color=AQUA, linewidth=1.2, label="B1 (validare)")
    ax.axhline(floor, color=VIOLET, linewidth=1.2, linestyle="-.", label=f"plafon de zgomot ≈ {floor:.2f}")
    for n, r in zip(ns, lc):
        ax.annotate(f"{r['cv_mae']:.2f}", (n, r["cv_max"]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color=INK2,
                    bbox=dict(facecolor=SURF, edgecolor="none", pad=1.0))
    ax.set_xlabel("stații de antrenare")
    ax.set_ylabel("MAE (puncte ocupate)")
    ax.set_title("B. Curba de învățare (modelul final)", loc="left")
    ax.set_ylim(0, 2.2)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[2]
    names = ["B0", "B1", "model anterior (monoton doar pe putere)", "model final (monoton)"]
    labels = ["B0\nprofil oraș", "B1\nocupare × puncte", "model anterior", "model final"]
    w = 0.26
    xs = np.arange(len(names))
    for i, (part, col, lab) in enumerate((("train", "#b7d3f6", "antrenare"), ("cv", BLUE, "CV (zone nevăzute)"),
                                           ("test", "#184f95", "test"))):
        vals = [out["bars"][n][part] for n in names]
        bars = ax.bar(xs + (i - 1) * w, vals, width=w - 0.03, color=col, label=lab)
        for bx, v in zip(bars, vals):
            ax.text(bx.get_x() + bx.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(floor, color=VIOLET, linewidth=1.2, linestyle="-.", label=f"plafon de zgomot ≈ {floor:.2f}")
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylabel("MAE (puncte ocupate)")
    ax.set_title("C. Antrenare vs CV vs test", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)

    fig.suptitle("Diagnostic: overfitting (distanța antrenare–validare) și underfitting (distanța până la plafon)",
                 x=0.01, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "fit_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    (C.REPORTS / "fit_diagnostics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
