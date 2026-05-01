import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

V_COLS = [f"U{i}_kV" for i in range(1, 5)]
T_COLS = [f"T{i}_s" for i in range(1, 5)]
INLET_COLS = ["Temp_C", "C_in_gNm3", "Q_Nm3h"]
TARGET = "C_out_mgNm3"
POWER = "P_total_kW"


@dataclass
class TypicalCondition:
    cluster_id: int
    size: int
    temp_mean: float
    cin_mean: float
    q_mean: float
    best_score: float
    best_energy_kwh: float
    predicted_cout: float
    voltage_plan: str
    rapping_plan: str


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for t in T_COLS:
        out[f"inv_{t}"] = 1.0 / np.clip(out[t], 1, None)
    out["hour"] = out["timestamp"].dt.hour
    out["dow"] = out["timestamp"].dt.dayofweek
    return out


def train_model(df_feat: pd.DataFrame):
    feats = INLET_COLS + V_COLS + T_COLS + [f"inv_{t}" for t in T_COLS] + ["hour", "dow"]
    X = df_feat[feats]
    y = df_feat[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pre = ColumnTransformer([
        ("num", StandardScaler(), feats)
    ])
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=3))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
    }
    return model, feats, metrics, X_test, y_test


def analyze_rapping_peaks(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for tcol in T_COLS:
        short = df[df[tcol] <= df[tcol].quantile(0.25)]
        long_ = df[df[tcol] >= df[tcol].quantile(0.75)]
        out.append({
            "field": tcol,
            "short_cycle_mean_peak_95": short[TARGET].quantile(0.95),
            "long_cycle_mean_peak_95": long_[TARGET].quantile(0.95),
            "difference": short[TARGET].quantile(0.95) - long_[TARGET].quantile(0.95),
        })
    return pd.DataFrame(out)


def cluster_conditions(df: pd.DataFrame, k: int = 4):
    X = df[["Temp_C", "C_in_gNm3"]]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    return labels


def search_optimal_for_cluster(df, model, feats, cluster_id, target_limit=10.0, n_grid=4):
    sub = df[df["cluster"] == cluster_id].copy()
    base = sub.iloc[0:1].copy()

    v_ranges = [np.linspace(sub[c].quantile(0.2), sub[c].quantile(0.9), n_grid) for c in V_COLS]
    t_ranges = [np.linspace(sub[c].quantile(0.2), sub[c].quantile(0.9), n_grid) for c in T_COLS]

    best = None
    for u1 in v_ranges[0]:
        for u2 in v_ranges[1]:
            for u3 in v_ranges[2]:
                for u4 in v_ranges[3]:
                    for t1 in t_ranges[0]:
                        for t2 in t_ranges[1]:
                            for t3 in t_ranges[2]:
                                for t4 in t_ranges[3]:
                                    row = base.copy()
                                    row[V_COLS] = [u1, u2, u3, u4]
                                    row[T_COLS] = [t1, t2, t3, t4]
                                    for t in T_COLS:
                                        row[f"inv_{t}"] = 1.0 / max(row[t].iloc[0], 1)
                                    row["hour"] = int(sub["timestamp"].dt.hour.mode().iloc[0])
                                    row["dow"] = int(sub["timestamp"].dt.dayofweek.mode().iloc[0])
                                    pred_c = float(model.predict(row[feats])[0])
                                    # 近似能耗模型：按电压平方与振打频率线性叠加
                                    e_v = 0.02 * (u1**2 + u2**2 + u3**2 + u4**2)
                                    e_t = 25.0 * (1/t1 + 1/t2 + 1/t3 + 1/t4)
                                    energy = e_v + e_t
                                    if pred_c <= target_limit:
                                        score = energy
                                        if (best is None) or (score < best["score"]):
                                            best = {
                                                "score": score,
                                                "energy": energy,
                                                "pred_c": pred_c,
                                                "U": [u1, u2, u3, u4],
                                                "T": [t1, t2, t3, t4],
                                            }
    if best is None:
        return None
    return best


def main(data_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_data(Path(data_path))
    dff = build_features(df)

    model, feats, metrics, X_test, y_test = train_model(dff)

    perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(out_dir / "q1_feature_importance.csv", index=False)

    peaks = analyze_rapping_peaks(df)
    peaks.to_csv(out_dir / "q1_rapping_peak_effect.csv", index=False)

    dff["cluster"] = cluster_conditions(dff, k=4)
    summary = dff.groupby("cluster")[INLET_COLS].agg(["mean", "std", "count"])
    summary.to_csv(out_dir / "q2_cluster_summary.csv")

    rows = []
    for c in sorted(dff["cluster"].unique()):
        best = search_optimal_for_cluster(dff, model, feats, c, target_limit=10.0)
        sub = dff[dff["cluster"] == c]
        if best is None:
            continue
        rows.append(TypicalCondition(
            cluster_id=int(c),
            size=int(len(sub)),
            temp_mean=float(sub["Temp_C"].mean()),
            cin_mean=float(sub["C_in_gNm3"].mean()),
            q_mean=float(sub["Q_Nm3h"].mean()),
            best_score=float(best["score"]),
            best_energy_kwh=float(best["energy"]),
            predicted_cout=float(best["pred_c"]),
            voltage_plan=",".join(f"{x:.2f}" for x in best["U"]),
            rapping_plan=",".join(f"{x:.1f}" for x in best["T"]),
        ))
    plan_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("cluster_id")
    plan_df.to_csv(out_dir / "q2_optimal_plan_10mg.csv", index=False)

    # Q4: tighten to 5 mg/Nm3
    rows5 = []
    for c in sorted(dff["cluster"].unique()):
        best10 = search_optimal_for_cluster(dff, model, feats, c, target_limit=10.0)
        best5 = search_optimal_for_cluster(dff, model, feats, c, target_limit=5.0)
        if (best10 is None) or (best5 is None):
            continue
        inc = (best5["energy"] - best10["energy"]) / best10["energy"] * 100
        rows5.append({"cluster": int(c), "energy_10": best10["energy"], "energy_5": best5["energy"], "increase_pct": inc})
    q4_df = pd.DataFrame(rows5)
    q4_df.to_csv(out_dir / "q4_energy_increase_5mg.csv", index=False)

    with open(out_dir / "model_report.txt", "w", encoding="utf-8") as f:
        f.write("=== 问题1：关系分析（随机森林）===\n")
        f.write(f"R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}\n")
        f.write("主要特征见 q1_feature_importance.csv；振打瞬时峰值影响见 q1_rapping_peak_effect.csv\n\n")
        f.write("=== 问题2：典型工况划分 + 10mg/Nm3优化===\n")
        f.write("工况划分依据：Temp_C 与 C_in_gNm3的KMeans聚类（4类），结果见 q2_cluster_summary.csv\n")
        f.write("各类最优电压/振打组合见 q2_optimal_plan_10mg.csv\n\n")
        f.write("=== 问题3：选取2个差异明显工况===\n")
        if len(plan_df) >= 2:
            hi = plan_df.sort_values("cin_mean", ascending=False).iloc[0]
            lo = plan_df.sort_values("cin_mean", ascending=True).iloc[0]
            f.write(f"高浓度工况 cluster={int(hi.cluster_id)}: U=[{hi.voltage_plan}] T=[{hi.rapping_plan}]\n")
            f.write(f"低浓度工况 cluster={int(lo.cluster_id)}: U=[{lo.voltage_plan}] T=[{lo.rapping_plan}]\n")
            f.write("策略差异解释：高浓度通常需优先提高前级电场电压并缩短振打周期；低浓度可适度降压并延长振打周期。\n\n")
        f.write("=== 问题4：5mg/Nm3标准影响===\n")
        if not q4_df.empty:
            f.write(f"平均能耗增幅: {q4_df['increase_pct'].mean():.2f}%\n")
            f.write("建议：高浓度工况采用分级电压提升+前场更高振打频率，避免单纯加压造成反电晕与无效能耗。\n")

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cement ESP协同优化建模")
    parser.add_argument("--data", default="Cement_ESP_Data.csv", help="输入CSV路径")
    parser.add_argument("--out", default="outputs", help="输出目录")
    args = parser.parse_args()
    main(args.data, args.out)
