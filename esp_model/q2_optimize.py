import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from esp_model.common import read_data, add_features, ensure_dir, INLET_COLS, V_COLS, T_COLS, TARGET


def fit_model(dff, feats):
    X = dff[feats]
    y = dff[TARGET]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Pipeline([
        ("pre", ColumnTransformer([("num", StandardScaler(), feats)])),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1, min_samples_leaf=2))
    ])
    model.fit(X_train, y_train)
    return model


def search_cluster(sub, model, feats, limit, grid_n):
    base = sub.iloc[0:1].copy()
    vr = [np.linspace(sub[c].quantile(0.2), sub[c].quantile(0.9), grid_n) for c in V_COLS]
    tr = [np.linspace(sub[c].quantile(0.2), sub[c].quantile(0.9), grid_n) for c in T_COLS]
    best = None
    for u1 in vr[0]:
        for u2 in vr[1]:
            for u3 in vr[2]:
                for u4 in vr[3]:
                    for t1 in tr[0]:
                        for t2 in tr[1]:
                            for t3 in tr[2]:
                                for t4 in tr[3]:
                                    row = base.copy()
                                    row[V_COLS] = [u1, u2, u3, u4]
                                    row[T_COLS] = [t1, t2, t3, t4]
                                    for t in T_COLS:
                                        row[f"inv_{t}"] = 1.0 / max(row[t].iloc[0], 1)
                                    row["hour"] = int(sub["timestamp"].dt.hour.mode().iloc[0])
                                    row["dow"] = int(sub["timestamp"].dt.dayofweek.mode().iloc[0])
                                    c = float(model.predict(row[feats])[0])
                                    e = 0.02 * (u1*u1 + u2*u2 + u3*u3 + u4*u4) + 25.0 * (1/t1 + 1/t2 + 1/t3 + 1/t4)
                                    if c <= limit:
                                        if best is None or e < best["energy"]:
                                            best = {"energy": e, "cout": c, "U": [u1,u2,u3,u4], "T":[t1,t2,t3,t4]}
    return best


def run(data_path: str, out_dir: str, k: int, grid_n: int):
    ensure_dir(out_dir)
    dff = add_features(read_data(data_path))
    feats = INLET_COLS + V_COLS + T_COLS + [f"inv_{t}" for t in T_COLS] + ["hour", "dow"]
    model = fit_model(dff, feats)
    kmx = StandardScaler().fit_transform(dff[["Temp_C", "C_in_gNm3"]])
    dff["cluster"] = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(kmx)
    dff.groupby("cluster")[INLET_COLS].agg(["mean", "std", "count"]).to_csv(f"{out_dir}/q2_cluster_summary.csv")
    rows = []
    for c in sorted(dff["cluster"].unique()):
        sub = dff[dff["cluster"] == c]
        best = search_cluster(sub, model, feats, limit=10.0, grid_n=grid_n)
        if best is None:
            continue
        rows.append({
            "cluster": int(c), "size": int(len(sub)), "temp_mean": sub["Temp_C"].mean(), "cin_mean": sub["C_in_gNm3"].mean(),
            "q_mean": sub["Q_Nm3h"].mean(), "pred_cout": best["cout"], "energy": best["energy"],
            "U1_kV": best["U"][0], "U2_kV": best["U"][1], "U3_kV": best["U"][2], "U4_kV": best["U"][3],
            "T1_s": best["T"][0], "T2_s": best["T"][1], "T3_s": best["T"][2], "T4_s": best["T"][3]
        })
    pd.DataFrame(rows).sort_values("cluster").to_csv(f"{out_dir}/q2_optimal_plan_10mg.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Cement_ESP_Data.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--grid", type=int, default=4)
    a = p.parse_args()
    run(a.data, a.out, a.k, a.grid)
