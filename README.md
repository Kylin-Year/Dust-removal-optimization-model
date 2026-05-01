# 水泥烧成系统电除尘器协同优化建模（分问题脚本）

## 0. 你先准备什么
把数据文件放在项目根目录，并命名为：

`Cement_ESP_Data.csv`

## 1. 一次性安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib
```

## 2. 一键跑完四问
```bash
python run_all.py --data Cement_ESP_Data.csv --out outputs --k 4 --grid 4
```

## 3. 分开跑每一问
```bash
python esp_model/q1_analysis.py --data Cement_ESP_Data.csv --out outputs
python esp_model/q2_optimize.py --data Cement_ESP_Data.csv --out outputs --k 4 --grid 4
python esp_model/q3_compare.py --out outputs
python esp_model/q4_tighten_standard.py --data Cement_ESP_Data.csv --out outputs --k 4 --grid 4
```

## 4. 每个问题看哪个文件

### 问题1
看：
- `outputs/q1_metrics.csv`
- `outputs/q1_feature_importance.csv`
- `outputs/q1_rapping_peak_effect.csv`
图：
- `outputs/q1_feature_importance_top12.png`
- `outputs/q1_rapping_peak_delta.png`

### 问题2
看：
- `outputs/q2_cluster_summary.csv`
- `outputs/q2_optimal_plan_10mg.csv`

### 问题3
看：
- `outputs/q3_two_conditions_table.csv`
图：
- `outputs/q3_voltage_compare.png`
- `outputs/q3_rapping_compare.png`

### 问题4
看：
- `outputs/q4_energy_increase_5mg.csv`
- `outputs/q4_summary.csv`

## 5. 你怎么打开结果
CSV文件可以用 Excel 直接打开。
PNG 图片双击即可查看。

## 6. 参数含义
- `--k`：典型工况聚类数量
- `--grid`：优化搜索网格密度，越大越慢但更细
