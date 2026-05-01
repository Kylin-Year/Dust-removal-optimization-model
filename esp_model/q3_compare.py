import argparse
import pandas as pd
import matplotlib.pyplot as plt
from esp_model.common import ensure_dir


def run(out_dir: str):
    ensure_dir(out_dir)
    plan = pd.read_csv(f"{out_dir}/q2_optimal_plan_10mg.csv")
    high = plan.sort_values("cin_mean", ascending=False).iloc[0]
    low = plan.sort_values("cin_mean", ascending=True).iloc[0]
    pd.DataFrame([high, low]).to_csv(f"{out_dir}/q3_two_conditions_table.csv", index=False)
    u_cols = ["U1_kV", "U2_kV", "U3_kV", "U4_kV"]
    t_cols = ["T1_s", "T2_s", "T3_s", "T4_s"]
    plt.figure(figsize=(8,4))
    x = range(4)
    plt.plot(x, [high[c] for c in u_cols], marker='o', label='high_cin')
    plt.plot(x, [low[c] for c in u_cols], marker='o', label='low_cin')
    plt.xticks(list(x), u_cols)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/q3_voltage_compare.png", dpi=200)
    plt.close()
    plt.figure(figsize=(8,4))
    plt.plot(x, [high[c] for c in t_cols], marker='o', label='high_cin')
    plt.plot(x, [low[c] for c in t_cols], marker='o', label='low_cin')
    plt.xticks(list(x), t_cols)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/q3_rapping_compare.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    run(a.out)
