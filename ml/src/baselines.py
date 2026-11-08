"""Baseline-uri, cu statisticile calculate doar pe train.

B0: media lui y per (hour, is_weekend) -> profilul mediu al orașului.
B1: rata medie de ocupare y / n_piles per (hour, is_weekend), înmulțită cu n_piles al stației.
"""
import pandas as pd

KEYS = ["hour", "is_weekend"]


class B0:
    name = "B0_city_profile"

    def fit(self, train: pd.DataFrame):
        self.profile_ = train.groupby(KEYS)["y"].mean().rename("pred")
        return self

    def predict(self, df: pd.DataFrame):
        return df[KEYS].join(self.profile_, on=KEYS)["pred"].to_numpy()


class B1:
    name = "B1_occupancy_x_piles"

    def fit(self, train: pd.DataFrame):
        occ = train["y"] / train["n_piles"]
        self.rate_ = occ.groupby([train["hour"], train["is_weekend"]]).mean().rename("rate")
        return self

    def predict(self, df: pd.DataFrame):
        return (df[KEYS].join(self.rate_, on=KEYS)["rate"] * df["n_piles"]).to_numpy()
