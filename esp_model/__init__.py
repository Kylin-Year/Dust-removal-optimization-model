from pathlib import Path
import numpy as np
import pandas as pd

V_COLS = [f"U{i}_kV" for i in range(1, 5)]
T_COLS = [f"T{i}_s" for i in range(1, 5)]
INLET_COLS = ["Temp_C", "C_in_gNm3", "Q_Nm3h"]
TARGET = "C_out_mgNm3"


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for t in T_COLS:
        out[f"inv_{t}"] = 1.0 / np.clip(out[t], 1, None)
    out["hour"] = out["timestamp"].dt.hour
    out["dow"] = out["timestamp"].dt.dayofweek
    return out


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
