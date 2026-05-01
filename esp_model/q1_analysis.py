import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from esp_model.common import read_data, add_features, ensure_dir, INLET_COLS, V_COLS, T_COLS, TARGET


def run(data_path: str, out_dir: str):
    ensure_dir(out_dir)
    df = read_data(data_path)
    dff = add_features(df)
    feats = INLET_COLS + V_COLS + T_COLS + [f"inv_{t}" for t in T_COLS] + ["hour", "dow"]
    X = dff[feats]
    y = dff[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Pipeline([
        ("pre", ColumnTransformer([("num", StandardScaler(), feats)])),
        ("rf", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1, min_samples_leaf=2))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pd.DataFrame([{
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5
    }]).to_csv(f"{out_dir}/q1_metrics.csv", index=False)
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(f"{out_dir}/q1_feature_importance.csv", index=False)
    rows = []
    for t in T_COLS:
        short = df[df[t] <= df[t].quantile(0.25)][TARGET].quantile(0.95)
        long = df[df[t] >= df[t].quantile(0.75)][TARGET].quantile(0.95)
        rows.append({"field": t, "peak95_short": short, "peak95_long": long, "delta_short_minus_long": short - long})
    peak = pd.DataFrame(rows)
    peak.to_csv(f"{out_dir}/q1_rapping_peak_effect.csv", index=False)
    plt.figure(figsize=(10, 5))
    top = imp.head(12).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/q1_feature_importance_top12.png", dpi=200)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.bar(peak["field"], peak["delta_short_minus_long"])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/q1_rapping_peak_delta.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Cement_ESP_Data.csv")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    run(a.data, a.out)
