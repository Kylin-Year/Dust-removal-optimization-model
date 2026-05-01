import argparse
from esp_model.q1_analysis import run as run_q1
from esp_model.q2_optimize import run as run_q2
from esp_model.q3_compare import run as run_q3
from esp_model.q4_tighten_standard import run as run_q4

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Cement_ESP_Data.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--grid", type=int, default=4)
    a = p.parse_args()
    run_q1(a.data, a.out)
    run_q2(a.data, a.out, a.k, a.grid)
    run_q3(a.out)
    run_q4(a.data, a.out, a.k, a.grid)
