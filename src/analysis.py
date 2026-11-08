"""Analiza modelului pe setul de test: erori, SHAP, ICE/PD, saturare.

Ieșiri: reports/figures/*.png, reports/analysis.json, reports/top20_stations.csv
Rulează după python -m src.evaluate (folosește reports/test_predictions.parquet).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.inspection import partial_dependence  # noqa: E402
from sklearn.metrics import precision_score, recall_score, roc_auc_score  # noqa: E402

from src import config as C  # noqa: E402
from src.evaluate import daily_hours  # noqa: E402
from src.train import load_model, predict_y  # noqa: E402

# paletă (categorial: albastru / portocaliu / aqua), suprafață și text
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 110,
})


class BoosterRegressor(RegressorMixin, BaseEstimator):
    """Wrapper minim peste lgb.Booster, ca sklearn.inspection să-l accepte ca estimator antrenat."""

    def __init__(self, booster=None, features=None, target_mode="y"):
        self.booster = booster
        self.features = features
        self.target_mode = target_mode

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.features) if not isinstance(X, pd.DataFrame) else X
        return predict_y(self.booster, X, self.features, self.target_mode)


def fmt_num(v: float) -> str:
    return f"{v / 1000:.0f}k" if abs(v) >= 10_000 else f"{v:.4g}"


def save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / name, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. reziduuri pe grupuri
def residuals_by_group(pred: pd.DataFrame) -> dict:
    pred = pred.assign(res=pred["pred_LGBM"] - pred["y"], abs_res=(pred["pred_LGBM"] - pred["y"]).abs(),
                       abs_res_b1=(pred["pred_B1"] - pred["y"]).abs())
    out = {}

    def summarize(g):
        return pd.Series({"n_rows": len(g), "n_stations": g["station_id"].nunique(),
                          "y_mean": g["y"].mean(), "MAE": g["abs_res"].mean(),
                          "MAE_B1": g["abs_res_b1"].mean(), "bias": g["res"].mean(),
                          "rel_MAE": g["abs_res"].mean() / g["y"].mean()})

    groups = [c for c in ["n_piles", "power_kw", "pop_1km", "n_stations_1km"] if pred[c].notna().all()]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 3.6), sharey=True)
    for ax, col in zip(axes, groups):
        # cvartile la nivel de stație (fiecare stație cântărește la fel)
        st = pred.drop_duplicates("station_id").set_index("station_id")[col]
        q = pd.qcut(st.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        edges = st.groupby(q, observed=True).agg(["min", "max"])
        g = pred.assign(q=pred["station_id"].map(q)).groupby("q", observed=True)[list(pred.columns)].apply(summarize)
        g["range"] = [f"{fmt_num(edges.loc[i, 'min'])}–{fmt_num(edges.loc[i, 'max'])}" for i in g.index]
        out[col] = g.reset_index().to_dict(orient="records")
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["MAE_B1"], width=0.38, color=AQUA, label="B1")
        ax.bar(x + 0.2, g["MAE"], width=0.38, color=ORANGE, label="LightGBM")
        ax.set_xticks(x, [f"{i}\n{r}" for i, r in zip(g.index, g["range"])], fontsize=8)
        ax.set_title(col)
    axes[0].set_ylabel("MAE (puncte ocupate)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Eroarea pe cvartile (la nivel de stație), set de test", x=0.01, y=1.04, ha="left", fontweight="bold")
    save(fig, "residuals_by_quartile.png")

    for col in ["hour", "is_weekend"]:
        g = pred.groupby(col)[list(pred.columns)].apply(summarize)
        out[col] = g.reset_index().to_dict(orient="records")
    gh = pred.groupby(["is_weekend", "hour"])[["abs_res", "abs_res_b1", "res", "y"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, (we, g) in zip(axes, gh.groupby("is_weekend")):
        ax.plot(g["hour"], g["abs_res_b1"], color=AQUA, label="MAE B1")
        ax.plot(g["hour"], g["abs_res"], color=ORANGE, label="MAE LightGBM")
        ax.plot(g["hour"], g["res"], color=VIOLET, linestyle="--", label="bias LightGBM")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title("weekend" if we else "zi lucrătoare")
        ax.set_xlabel("ora")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("puncte ocupate")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    save(fig, "residuals_by_hour.png")
    return out


# ---------------------------------------------------------------- 2. top 20 stații
def top_stations(pred: pd.DataFrame, feats: pd.DataFrame, stats: pd.DataFrame) -> dict:
    real = daily_hours(pred, "y").rename("daily_real")
    p = daily_hours(pred, "pred_LGBM").rename("daily_pred")
    st = pd.concat([real, p], axis=1)
    st["err"] = st["daily_pred"] - st["daily_real"]
    st["abs_err"] = st["err"].abs()
    st["rel_err"] = st["err"] / st["daily_real"].clip(lower=1)
    cols = ["n_piles", "power_kw", "power_imputed", *C.COMPETITION_FEATURES, *C.CONTEXT_FEATURES]
    st = st.join(feats.set_index("station_id")[cols]).join(
        stats.set_index("station_id")[["coverage", "sat_share"]])
    st["occ_real"] = st["daily_real"] / 24 / st["n_piles"]
    st["occ_pred"] = st["daily_pred"] / 24 / st["n_piles"]
    top = st.sort_values("abs_err", ascending=False).head(20)
    top.round(3).to_csv(C.REPORTS / "top20_stations.csv")
    comp = pd.DataFrame({"top20_median": top[cols + ["occ_real", "occ_pred", "sat_share", "coverage"]].median(),
                         "test_median": st[cols + ["occ_real", "occ_pred", "sat_share", "coverage"]].median()})
    # erorile relative mari (stații mici/goale) sunt alt tip de caz decât cele absolute
    top_rel = st[st["daily_real"] >= 1].sort_values("rel_err", key=np.abs, ascending=False).head(20)
    return {
        "station_level": st.reset_index(),
        "summary": {
            "n_top": 20,
            "n_overpredicted": int((top["err"] > 0).sum()),
            "n_underpredicted": int((top["err"] < 0).sum()),
            "share_of_total_abs_error": float(top["abs_err"].sum() / st["abs_err"].sum()),
            "feature_medians_top20_vs_test": comp.round(3).to_dict(orient="index"),
            "top20_by_relative_error": {
                "n_overpredicted": int((top_rel["err"] > 0).sum()),
                "median_daily_real": float(top_rel["daily_real"].median()),
                "median_occ_real": float(top_rel["occ_real"].median()),
                "median_occ_pred": float(top_rel["occ_pred"].median()),
            },
        },
    }


# ---------------------------------------------------------------- 3. profiluri orare
def profiles(pred: pd.DataFrame, st: pd.DataFrame) -> dict:
    st = st.set_index("station_id")
    typical = st.iloc[(st["abs_err"] - st["abs_err"].median()).abs().argsort()[:3]].index.tolist()
    atypical = st.sort_values("abs_err", ascending=False).head(3).index.tolist()
    sel = typical + atypical
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for ax, sid in zip(axes.ravel(), sel):
        d = pred[(pred["station_id"] == sid) & (pred["is_weekend"] == 0)].sort_values("hour")
        ax.plot(d["hour"], d["y"], color=BLUE, label="real")
        ax.plot(d["hour"], d["pred_LGBM"], color=ORANGE, label="LightGBM")
        ax.plot(d["hour"], d["pred_B1"], color=AQUA, linestyle="--", label="B1")
        ax.axhline(d["n_piles"].iloc[0], color=INK2, linewidth=0.8, linestyle=":")
        kind = "tipică" if sid in typical else "atipică"
        ax.set_title(f"stația {sid} ({kind}), {int(d['n_piles'].iloc[0])} puncte", fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("puncte ocupate (zi lucrătoare)")
    for ax in axes[1]:
        ax.set_xlabel("ora")
    fig.suptitle("Profil orar real vs prezis (linia punctată = numărul de puncte)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "profiles_typical_atypical.png")
    return {"typical": [int(s) for s in typical], "atypical": [int(s) for s in atypical]}


# ---------------------------------------------------------------- 4. SHAP
def shap_analysis(booster, features, X: pd.DataFrame, target_mode: str) -> dict:
    sv = shap.TreeExplainer(booster).shap_values(X[features])
    fig = plt.figure()
    shap.summary_plot(sv, X[features], show=False, max_display=len(features))
    plt.title("SHAP pe setul de test (ieșirea brută a modelului)", loc="left")
    save(plt.gcf(), "shap_summary.png")
    plt.close(fig)

    imp = pd.Series(np.abs(sv).mean(0), index=features).sort_values(ascending=False)
    groups = {
        "competiție": C.COMPETITION_FEATURES,
        "populație și POI": [f for f in C.CONTEXT_FEATURES if f in features],
        "stație (n_piles, power_kw)": C.STATION_FEATURES,
        "timp (hour, is_weekend)": C.TEMPORAL_FEATURES,
    }
    idx = {f: i for i, f in enumerate(features)}
    # importanța de grup = media lui |suma SHAP pe grup| (SHAP e aditiv, deci se poate însuma per rând)
    grouped = {g: float(np.abs(sv[:, [idx[f] for f in fs]].sum(1)).mean()) for g, fs in groups.items() if fs}
    grouped = dict(sorted(grouped.items(), key=lambda kv: -kv[1]))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(list(grouped)[::-1], list(grouped.values())[::-1], color=BLUE, height=0.6)
    for i, v in enumerate(list(grouped.values())[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("medie |Σ SHAP pe grup|")
    ax.set_title("Importanța grupată a feature-urilor", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "shap_grouped.png")
    return {"mean_abs_shap": imp.round(4).to_dict(), "grouped_mean_abs_shap": grouped,
            "competition_sum_of_individual": float(imp[C.COMPETITION_FEATURES].sum()),
            "explained_output": ("rata de ocupare (ieșirea brută a modelului; pentru cross_entropy/tweedie e pe "
                                 "scara log-odds/log)") if target_mode == "occ" else
                                "y (ieșirea brută; pentru tweedie e pe scara log)"}


# ---------------------------------------------------------------- 5. ICE / partial dependence
def ice_checks(booster, features, X: pd.DataFrame, target_mode: str) -> dict:
    est = BoosterRegressor(booster, features, target_mode)
    sample = X[features].sample(n=min(1500, len(X)), random_state=C.SEED).astype(float)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, feat in zip(axes, ["n_piles", "power_kw"]):
        grid_vals = np.unique(np.quantile(X[feat], np.linspace(0, 0.99, 40)))
        pdp = partial_dependence(est, sample, [feat], kind="both", custom_values={feat: grid_vals})
        grid, avg, ind = pdp["grid_values"][0], pdp["average"][0], pdp["individual"][0]
        rng = np.random.default_rng(C.SEED)
        for row in ind[rng.choice(len(ind), size=120, replace=False)]:
            ax.plot(grid, row, color=BLUE, alpha=0.12, linewidth=0.8)
        ax.plot(grid, avg, color=ORANGE, linewidth=2.5, label="medie (PD)")
        ax.set_xlabel(feat)
        ax.set_ylabel("cerere prezisă (puncte ocupate)")
        ax.set_title(f"ICE / PD: {feat}", loc="left")
        ax.legend(loc="upper left")
        if feat == "power_kw":
            ax.set_xscale("log")
        d = np.diff(ind, axis=1)
        steps = np.diff(avg)
        half = len(grid) // 2
        out[feat] = {
            "grid": grid.round(2).tolist(), "pd": avg.round(4).tolist(),
            "pd_monotone_increasing": bool(np.all(steps >= -1e-9)),
            "pd_monotone_decreasing": bool(np.all(steps <= 1e-9)),
            "share_ice_curves_nondecreasing": float(np.mean(np.all(d >= -1e-9, axis=1))),
            "share_ice_curves_nonincreasing": float(np.mean(np.all(d <= 1e-9, axis=1))),
            "pd_slope_low_half": float((avg[half] - avg[0]) / (grid[half] - grid[0])),
            "pd_slope_high_half": float((avg[-1] - avg[half]) / (grid[-1] - grid[half])),
        }
    fig.tight_layout()
    save(fig, "ice_n_piles_power.png")

    # aceeași curbă exprimată ca rată de ocupare: cererea pe punct scade dacă efectul se saturează
    g = np.array(out["n_piles"]["grid"])
    out["n_piles"]["pd_occupancy_per_pile"] = (np.array(out["n_piles"]["pd"]) / g).round(4).tolist()
    return out


# ---------------------------------------------------------------- 6. saturare
def saturation(pred: pd.DataFrame, stats: pd.DataFrame) -> dict:
    p = pred.assign(occ_pred=pred["pred_LGBM"] / pred["n_piles"], occ_real=pred["y"] / pred["n_piles"])
    st = p.groupby("station_id").agg(peak_occ_pred=("occ_pred", "max"), peak_occ_real=("occ_real", "max"),
                                      n_piles=("n_piles", "first"))
    st = st.join(stats.set_index("station_id")["sat_share"])
    st["sat_real"] = st["sat_share"] >= C.SAT_SHARE_REAL
    st["sat_real_peak"] = st["peak_occ_real"] > C.SAT_OCC
    st["sat_pred"] = st["peak_occ_pred"] > C.SAT_OCC

    def pr(real, predicted):
        k = int(real.sum())
        top_k = st["peak_occ_pred"].nlargest(k).index if k else []
        return {"n_real_positive": k, "n_pred_positive": int(predicted.sum()),
                "precision": float(precision_score(real, predicted, zero_division=0)),
                "recall": float(recall_score(real, predicted, zero_division=0)),
                # modelul prezice medii, deci rareori atinge pragul; clasarea e informația utilă
                "precision_at_k_top_predicted": float(real.loc[top_k].mean()) if k else None,
                "base_rate": float(real.mean()),
                "roc_auc_of_peak_occ_pred": float(roc_auc_score(real, st["peak_occ_pred"]))
                if 0 < real.sum() < len(real) else None}

    res = {
        "n_stations": int(len(st)),
        "definition": {
            "real": f"sat_share >= {C.SAT_SHARE_REAL} (procentul de intervale de 5 min cu toate punctele ocupate)",
            "real_peak": f"rata de ocupare reală la ora de vârf > {C.SAT_OCC} (comparabilă direct)",
            "pred": f"rata de ocupare prezisă la ora de vârf > {C.SAT_OCC}",
        },
        "vs_sat_share": pr(st["sat_real"], st["sat_pred"]),
        "vs_real_peak_occupancy": pr(st["sat_real_peak"], st["sat_pred"]),
        "spearman_peak_occ_pred_vs_sat_share": float(spearmanr(st["peak_occ_pred"], st["sat_share"]).statistic),
        "peak_occ_pred_quantiles": st["peak_occ_pred"].quantile([0.5, 0.9, 0.99, 1]).round(3).to_dict(),
        "peak_occ_real_quantiles": st["peak_occ_real"].quantile([0.5, 0.9, 0.99, 1]).round(3).to_dict(),
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(st["peak_occ_pred"], st["sat_share"], s=14, color=BLUE, alpha=0.6, edgecolor="#fcfcfb",
               linewidth=0.5)
    ax.axvline(C.SAT_OCC, color=ORANGE, linewidth=1, linestyle="--")
    ax.axhline(C.SAT_SHARE_REAL, color=ORANGE, linewidth=1, linestyle="--")
    ax.text(C.SAT_OCC, ax.get_ylim()[1], " prag prezis", color=INK2, va="top", fontsize=8)
    ax.text(ax.get_xlim()[0], C.SAT_SHARE_REAL, " prag real", color=INK2, va="bottom", fontsize=8)
    ax.set_xlabel("rata de ocupare prezisă la ora de vârf")
    ax.set_ylabel("% intervale cu toate punctele ocupate (real)")
    ax.set_title("Saturare: prezis vs real (stațiile din test)", loc="left")
    save(fig, "saturation.png")
    return res


def main():
    pred = pd.read_parquet(C.PREDICTIONS_FILE)
    booster, meta = load_model()
    features = meta["features"]
    feats = pd.read_parquet(C.STATION_FEATURES_FILE)
    stats = pd.read_parquet(C.STATION_STATS_FILE)

    out = {"model": {k: meta.get(k) for k in ("feature_set", "monotone", "target_mode", "num_boost_round")},
           "residuals_by_group": residuals_by_group(pred)}
    top = top_stations(pred, feats, stats)
    out["top20"] = top["summary"]
    out["profiles"] = profiles(pred, top["station_level"])
    out["shap"] = shap_analysis(booster, features, pred, meta.get("target_mode", "y"))
    out["ice"] = ice_checks(booster, features, pred, meta.get("target_mode", "y"))
    out["saturation"] = saturation(pred, stats)
    C.ANALYSIS_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))

    print("SHAP grupat:", json.dumps(out["shap"]["grouped_mean_abs_shap"], ensure_ascii=False))
    print("SHAP individual:", json.dumps(out["shap"]["mean_abs_shap"]))
    for f in ("n_piles", "power_kw"):
        i = out["ice"][f]
        print(f"ICE {f}: PD crescător={i['pd_monotone_increasing']}, descrescător={i['pd_monotone_decreasing']}, "
              f"ICE nedescrescătoare={i['share_ice_curves_nondecreasing']:.0%}, "
              f"pantă jumătatea de jos={i['pd_slope_low_half']:.4f}, de sus={i['pd_slope_high_half']:.4f}")
    print("Top 20:", json.dumps({k: v for k, v in out["top20"].items() if k != "feature_medians_top20_vs_test"}))
    print("Saturare:", json.dumps({k: out["saturation"][k] for k in
                                    ("vs_sat_share", "vs_real_peak_occupancy",
                                     "spearman_peak_occ_pred_vs_sat_share")}))


if __name__ == "__main__":
    main()
