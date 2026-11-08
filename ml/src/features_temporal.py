"""Feature-uri temporale: ora din zi (0-23) și is_weekend (sâmbătă/duminică)."""
import pandas as pd


def temporal_grid() -> pd.DataFrame:
    """Toate cele 48 de combinații (hour, is_weekend)."""
    return pd.MultiIndex.from_product([range(24), (0, 1)], names=["hour", "is_weekend"]).to_frame(index=False)


def add_temporal(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    t = pd.to_datetime(df[time_col])
    return df.assign(hour=t.dt.hour, is_weekend=(t.dt.dayofweek >= 5).astype(int))
