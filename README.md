# 1. 项目简介
ret_pred 是一个面向股票收益预测的工程化量化建模框架，旨在构建一个可扩展、可复现、完全由配置驱动的研究系统。框架围绕 long-format 数据结构设计，支持从数据加载、预处理、滚动切分、模型训练、自动调参到评估与推理的完整流程，并通过 YAML 配置统一管理所有实验参数与路径。
主要功能包括：
支持 long-format 数据输入（date, stockid, y, features）
支持滚动窗口（walk-forward）训练
支持树模型（LightGBM / XGBoost / CatBoost）
支持深度学习模型（Linear / LSTM / GRU / Transformer）
支持回归任务和分类任务
训练阶段支持手动输入参数合集训练，也支持自动调参、grid search 等调参策略，能够根据指定指标进行参数搜索与性能门槛控制
所有模型文件、预测结果与评估指标均按实验编号自动落盘保存
支持评估指标计算与可视化
训练完成后还可直接进入推理阶段（（post-train inference）），对指定日期生成预测信号，适用于实盘前信号生成或周期性更新场景。
该框架适合用于因子收益预测、Alpha 模型研究、多模型对比实验以及策略前置信号构建等量化研究任务。


# 2. 目录结构
ret_pred/
│
├── main.py                # 项目入口（train / predict 调度）
├── paths.py               # 路径渲染与统一管理
├── dataloader.py          # long-format 数据加载
├── preprocess.py          # 数据清洗、缺失处理、winsorize、zscore
├── split.py               # 日期窗口切分
├── windows.py             # 生成 rolling windows
│
├── cut/                   # long_df -> 模型输入 payload
│   ├── tree.py
│   ├── seq.py
│   ├── dispatch.py
│   └── base.py
│
├── tree_models/           # 树模型实现
│   ├── lgbm.py
│   ├── xgb.py
│   ├── catboost.py
│   └── builder.py
│
├── nn_models/             # 深度学习模型
│   ├── linear.py
│   ├── lstm.py
│   ├── gru.py
│   ├── transformer.py
│   └── builder.py
│
├── losses/                # 可插拔损失函数系统
│   ├── regression.py
│   ├── classification.py
│   ├── objective_map.py
│   └── builder.py
│
├── trainer/               # 滚动训练核心模块
│   ├── rolling_trainer.py
│   ├── plugins.py
│   ├── model_bundle.py
│   ├── model_select.py
│   ├── builder.py
│   └── sweep_trainer.py
│
├── predictor/             # 滚动预测模块
│   └── rolling_predictor.py
│
├── evaluate/              # 评估与可视化
│   ├── evaluator.py
│   ├── tasks.py
│   ├── sweep_compare.py
│   └── builder.py
│
└── utils/
    └── logger.py

# 3. 安装指南
## 3.1 Python 环境
推荐：
Python 3.10+
macOS / Linux
建议使用虚拟环境：
python -m venv .venv
source .venv/bin/activate

## 3.2 安装依赖
最小依赖：
pip install numpy pandas pyyaml pyarrow matplotlib
pip install lightgbm xgboost catboost
pip install torch  # 如果使用深度学习模型
如果未来加入 requirements.txt，可直接：
pip install -r requirements.txt

# 4. 使用方法
项目入口为：
ret_pred/main.py
必须使用模块方式运行，而不是直接运行 main.py，以确保包内相对导入正确：
python -m ret_pred.main --config ret_pred/config_train.yaml
python -m ret_pred.main --config ret_pred/config_train_sweep.yaml
python -m ret_pred.main --config ret_pred/config_pred.yaml

## 4.1 rolling调参训练模式（Train）
运行方式如下：
python -m ret_pred.main --config ret_pred/config_train.yaml
在训练模式下，系统会首先读取 YAML 配置文件并初始化日志系统，然后加载 long-format 数据并执行完整的预处理流程，包括缺失值处理、去极值和标准化等操作。随后根据配置生成滚动窗口，对每个窗口构建模型输入数据并调用 RollingTrainer 进行训练与调参。训练完成后会自动保存模型与预测结果，并执行评估模块生成指标与可视化图表，所有产物均保存在对应的 run 目录下。

输出目录结构示例：
runs/
  exp001/
    logs/
    preprocess/
    model/
    preds/
    eval/

### 4.1.1 关键配置说明（config_train.yaml）
本框架的所有行为均由 YAML 配置驱动。每一个模块的逻辑、路径、模型、损失函数、调参方式以及评估策略都通过配置文件进行控制。下面对完整配置结构进行详细说明。

task:
  mode: "train"
  type: "regression"
  label_col: "y"
  seed: 42
该模块用于控制运行模式与任务类型。mode 用于指定当前运行是训练还是预测。train 模式会执行完整的数据加载、预处理、滚动训练、调参、模型保存与评估流程。predict 模式则只加载已训练模型进行推理，不构建标签也不执行评估。
type 指定任务类型，目前支持 regression 与 classification，会影响 loss 构建以及 evaluator 的评估逻辑。
label_col 指定标签列名。seed 为全局随机种子，用于保证实验可复现，包括 numpy、模型和调参过程。


logging:
  level: DEBUG
  log_dir: "{run_dir}/logs/train"
  filename: "train.log"
  max_bytes: 10485760
  backup_count: 100
  fmt: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  datefmt: "%Y-%m-%d %H:%M:%S"
  console: true
该模块用于配置日志系统。支持日志文件自动轮转，当文件大小超过 max_bytes 时自动创建新文件，最多保留 backup_count 个历史日志。日志内容包含时间、级别、模块名称和消息内容。console 为 true 时会同时输出到终端。


paths:
  runs_root: "/Users/yulia/Desktop/clickhouse/runs"
  run_id: "exp001"
  run_dir: "{runs_root}/{run_id}"
  preprocess_dir: "{run_dir}/preprocess"
  model_dir: "{run_dir}/model"
  pred_dir: "{run_dir}/preds"
  eval_dir: "{run_dir}/eval"
该模块用于统一实验目录结构管理。每个 run_id 代表一次完整实验。所有模型文件、预测结果、日志、预处理文件和评估结果都会自动保存在 runs_root/run_id 目录下。通过修改 run_id 即可创建新的实验版本，实现可复现的实验管理。


dataloader:
  source: "parquet_dir"
  date_col: "date"
  stockid_col: "stockid"
  long_filename: "xxx.parquet"
  label_name: "y"
  date_start: "2024-06-01"
  date_end: "2024-06-14"
  feature_mask_csv: "field_mask.csv"
  feature_col_name: "feature"
  feature_use_col: "use"
  build_label: true
  label_price_col: "open_1d"
  label_log_return: false
该模块负责加载 long-format 数据，并可自动构建标签。long-format 数据结构应包含 date、stockid、标签以及特征列。
feature_mask_csv 用于控制因子版本。系统会读取 CSV 文件，仅保留 use=1 的特征列，便于进行因子筛选与版本管理。
build_label 用于控制是否构建标签。在 train 模式下必须为 true，在 predict 模式下必须为 false。label_price_col 指定构建收益所用的价格列，例如 open_1d 表示 open-to-open 收益。label_log_return 用于控制是否构建对数收益。
date_start 与 date_end 用于控制数据区间。


preprocess:
  col_drop_threshold: 0.20
  row_drop_threshold: 0.20
  nan_policy: "tree_friendly"
  do_fill_for_tree: false
  fill_method: "ffill_then_zero"
  ffill_limit: 5
  winsorize:
    enabled: true
    by: "date"
    lower_q: 0.01
    upper_q: 0.99
  zscore:
    enabled: false
    by: "date"
    ddof: 0
    clip: null
  save_parquet: true
  save_state_json: true
该模块负责数据清洗与标准化处理。
col_drop_threshold 表示某特征缺失比例超过该阈值则删除该列。row_drop_threshold 表示某样本缺失比例超过阈值则删除该样本。
nan_policy 控制缺失值策略。strict 表示不允许存在任何 NaN，适用于线性模型或神经网络。tree_friendly 表示允许 NaN 存在，适用于 LightGBM、XGBoost 等树模型。
fill_method 为 ffill_then_zero 时表示每只股票最多向前填充 ffill_limit 天，剩余缺失填 0。
winsorize 用于去极值，通常按 date 横截面进行分位裁剪。zscore 用于标准化，可选择按横截面或全局标准化。
save_state_json 会保存预处理状态，包括删除列、填充策略和最终特征列表，用于推理阶段保持一致性。


datasplit:
  strategy: "step_then_ratio"
  window_ratio: 0.7
  min_window_dates: 5
  step_ratio: 0.07
  train_ratio: 0.7
  valid_ratio: 0.2
  test_ratio: 0.1
  return_dfs: false
该模块用于生成滚动窗口。
首先根据 window_ratio 和 min_window_dates 计算窗口长度，然后按照 step_ratio 滑动窗口。每个窗口内部再根据 train_ratio、valid_ratio 和 test_ratio 划分训练、验证和测试集。
return_dfs 为 false 表示使用 streaming 模式，不在内存中一次性生成所有窗口数据，而是在每个 step 动态读取预处理后的 parquet 数据，以节省内存。


datacutting:
  mode: "tree"
  seq_len: 60
  incomplete_policy: "drop"
该模块负责将 long-format 数据转换为模型输入格式。
mode 为 tree 时会构造二维特征矩阵。mode 为 seq 时会构造时间序列张量，用于 LSTM、GRU 或 Transformer。
seq_len 指定时间序列长度。incomplete_policy 用于控制历史长度不足时的处理方式，可选择丢弃样本或进行 padding。


model:
  family: "tree"
  name: "lgbm"
  candidates:
    lgbm:
      n_estimators: 800
      learning_rate: 0.05
该模块用于指定模型类型与默认参数。
family 用于区分 tree 或 nn 模型。name 指定具体模型。candidates 用于存储各模型默认参数以及调参候选空间。
本框架设计为不强制统一参数名，每个模型按照其原生参数名配置，builder 会自动通过 kwargs 方式传入模型构造函数。


trainer:
  name: "rolling"
  params:
    metric: "rankic"
    maximize: true
    loss:
      enabled: true
      name: "mse"
该模块为滚动训练核心配置。
metric 指定调参与模型更新的指标，例如 rankic、icir、mse、mae 或 r2。maximize 表示指标是否越大越好。
loss 用于指定训练损失函数。支持 mse、mae、huber、quantile 等。
tuner 模块用于控制调参方式，可选择 grid_search 或 random_search。update_gate 用于控制参数更新阈值，防止小幅波动导致频繁更新。
model_save 模块用于控制模型保存策略，例如保存最后一个模型或保存验证集最优模型。
saver 模块用于保存 train、valid、test 三个阶段的预测结果。


evaluate:
  name: "default"
  params:
    task: "regression"
    parts: ["train", "valid", "test"]
    regression:
      metrics: ["rankic", "icir", "mse", "mae", "r2"]
该模块用于从 preds 目录读取预测结果并计算指标，同时生成可视化图表。
支持计算 RankIC、ICIR、MSE、MAE、R2 等指标，并支持分位收益统计。

post_train_predict:
  enabled: true
  asof_date: "2024-06-14"
  target_date: "2024-06-17"
该模块用于在训练完成后自动执行一次推理，适用于例如周五训练后自动生成下周一预测信号的场景。

## 4.2 Sweep给定参数训练模式（Sweep_Train）

运行方式如下：
python -m ret_pred.main --config ret_pred/config_train_sweep.yaml
Sweep_Train 的目标是对一组“手工给定的参数集合”进行批量滚动训练对比。系统会先按配置加载 long-format 数据并执行一次完整预处理（缺失处理、去极值、标准化等），然后生成滚动窗口；
与 rolling 调参训练不同的是，Sweep_Train 不做 tuner 搜索，也不做 update_gate 的参数更新门槛判断，而是直接对 sweep.param_sets 中的每一组参数分别跑完整个 rolling 训练流程，并把每组参数的模型、预测、评估结果落到各自独立的目录中。训练结束后会生成 sweep_summary.parquet 汇总表，并运行sweep_compare 模块，将不同参数组在同一类图表上的表现拼接到一张对比图中，便于快速横向挑选最优参数风格。
输出目录结构上建议使用独立 run_id（例如 exp001_sweep），避免覆盖 rolling 训练的产物。Sweep 的所有单组参数实验会落在 runs/{run_id}/sweeps/{param_id}/ 下；全局汇总与对比图会落在 runs/{run_id}/sweeps/ 与 runs/{run_id}/eval/sweep_compare/ 下。

输出目录结构示例（每组参数一个子目录）：
runs/
│
└── exp001_sweep/                       # Sweep 实验根目录（run_id=exp001_sweep）
    │
    ├── logs/                           # 主控日志（sweep_rolling 全流程日志）
    │   └── train.log                   # 记录：每个 param_id 的训练/评估路径、汇总、报错堆栈等
    ├── preprocess/                       # 预处理缓存（一次预处理，多组参数复用）
    │   ├── preprocessed_train_y_2024-06-01_2024-06-14.parquet
    │   │                                 # 预处理后的 long 数据（缺失处理/去极值/标准化后结果）
    │   └── (可选) preprocess_state_fit_y_2024-06-01_2024-06-14.json
    │                    # 预处理状态（最终特征列表、drop 规则、填充策略等，用于复盘/推理对齐）
    │
    ├── sweeps/               # Sweep 每组参数的独立子实验目录（核心产物都在这里）
    │   ├── sweep_summary.parquet                         
    │   ├── lgbm_01/                    # 第 1 组参数（param_id = lgbm_01）
    │   │   ├── model/                  # 模型产物（该参数组的模型文件）
    │   │   │   └── model.pkl           # 按 trainer.model_save.strategy 保存（通常 last）
    │   │   ├── preds/                  # 预测落盘（该参数组所有窗口/折的预测）
    │   │   │   ├── step0_fold0_train.parquet    
    │   │   │   ├       # 训练集预测（含 y_true/y_pred/part/date/stockid/step_id/fold 等）
    │   │   │   ├── step0_fold0_test.parquet                # 测试集预测
    │   │   │   ├── step1_fold0_train.parquet
    │   │   │   ├── step1_fold0_test.parquet
    │   │   │   └── ...                                     # 直到所有 rolling step 跑完
    │   │   │
    │   │   └── eval/                                       # 单组参数评估产物（指标 + 单图）
    │   │       ├── metrics.csv         # 该参数组的指标汇总（按 part=train/test）
    │   │       ├── quantile_cum_return_bar.png            # 分位组合累计收益（分组柱状图）
    │   │       ├── quantile_mean_return.png               # 分位组合平均收益（折线）
    │   │       ├── rankic_distribution.png                # RankIC 分布直方图
    │   │       ├── residual_hist.png                      # 残差分布
    │   │       ├── residual_time_curve.png                # 残差随时间变化
    │   │       ├── scatter_true_vs_pred.png               # y_true vs y_pred 散点
    │   │       ├── time_curve_true_vs_pred.png            # 日度聚合 true/pred（同图）
    │   │       ├── time_curve_true_vs_pred_by_part.png    # 分 part 的 true/pred（多子图）
    │   │       └── (可选) regression_all_plots_long.png    # 不推荐直接生成长图
    │   │
    │   ├── lgbm_02/
    │   │   ├── model/
    │   │   │   └── model.pkl
    │   │   ├── preds/
    │   │   │   └── step{step_id}_fold{fold}_{part}.parquet
    │   │   └── eval/
    │   │       ├── metrics.csv
    │   │       └── *.png
    │   │
    │   ├── lgbm_03/
    │   │   └── ...
    │   └── lgbm_10/
    │       └── ...
    │
    └── eval/                                      # Sweep 全局评估目录（跨参数对比输出）
        └── sweep_compare/                           # sweep_compare 模块输出目录
            ├── metric_bar.png                       # 指标柱状图（按 sort_by 或指定指标排序）
            ├── compare_panel_01__test_quantile.png  # 同一种panel跨参数拼图
            ├── compare_panel_02__mean_return_by_quant



### 4.2.1 关键配置说明（config_train_sweep.yaml）

本框架的 sweep 训练同样由 YAML 配置完全驱动，但需要额外注意：
paths.run_id 建议单独命名以避免覆盖 rolling 实验；
split时valid设置为0，只保留train和test
trainer 需要切换为 sweep_rolling 并关闭 tuner/schedule；
evaluate必须保留单图（stitch_keep_individual=true），否则 sweep_compare 无法按“同类型图”进行横向拼接；
post_train_predict 一般关闭避免只对最后一组参数做推理导致结果歧义。下面按配置结构说明 sweep 训练的关键点。

config_train_sweep.yaml：

task:
mode: "train"
type: "regression"
label_col: "y"
seed: 42
该模块用于控制运行模式与任务类型。sweep 模式本质仍属于 train，会执行数据加载、预处理、滚动切窗、训练、保存与评估流程。type 指定任务类型（regression/classification）会影响 loss 与 evaluator 计算逻辑。label_col 指定标签列名。seed 为全局随机种子，保证每一组参数在相同随机环境下运行，便于公平对比。

logging:
level: INFO
log_dir: "{run_dir}/logs"
filename: "train.log"
max_bytes: 10485760
backup_count: 100
fmt: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
datefmt: "%Y-%m-%d %H:%M:%S"
console: true
日志系统与 rolling 模式一致，支持控制台与文件输出及轮转。建议 sweep 用 INFO 级别即可，避免 10 组参数产生过大日志。所有 sweep 的总控日志会写入 runs/{run_id}/logs/train.log，若你希望每组参数单独日志，应在 sweep trainer 内额外按 param_id 创建子日志目录。

paths:
runs_root: "/Users/yulia/Desktop/clickhouse/runs"
run_id: "exp001_sweep"
run_dir: "{runs_root}/{run_id}"
preprocess_dir: "{run_dir}/preprocess"
model_dir: "{run_dir}/model"
pred_dir: "{run_dir}/preds"
eval_dir: "{run_dir}/eval"
该模块用于统一实验目录结构管理。sweep 训练强烈建议使用独立 run_id，例如 exp001_sweep，避免覆盖 rolling 训练的 runs/exp001。注意：虽然 paths 里有 model_dir/pred_dir/eval_dir，但在 sweep 模式下每一组参数的产物通常会落在 {run_dir}/sweeps/{param_id}/... 的子目录中（由 trainer.model_save.out_dir 等配置控制）。eval_dir 仍用于保存 sweep_compare 等全局汇总类产物。

dataloader:
source: "parquet_dir"
date_col: "date"
stockid_col: "stockid"
long_filename: "/Users/yulia/Desktop/clickhouse/testdata_0101_0614.parquet"
fields: []
label_name: "y"
date_start: "2024-06-01"
date_end: "2024-06-14"
feature_mask_csv: "/Users/yulia/Desktop/clickhouse/field_mask.csv"
feature_col_name: "feature"
feature_use_col: "use"
build_label: true
label_price_col: "open_1d"
label_log_return: false
该模块负责加载 long-format 数据并构建标签。sweep 与 rolling 一样要求输入为 long-format（date, stockid, y, features...）。feature_mask_csv 用于因子版本控制，仅保留 use=1 的特征列，确保所有参数组在同一特征集合上训练。build_label 在 train 模式必须为 true；label_price_col 控制收益构造方式（例如 open_1d 表示 open-to-open），label_log_return 控制是否构造对数收益。date_start/date_end 控制数据区间，建议 sweep 全程固定区间以保证对比公平。

preprocess:
date_col: "date"
stockid_col: "stockid"
label_col: "y"
col_drop_threshold: 0.20
row_drop_threshold: 0.20
nan_policy: "tree_friendly"
do_fill_for_tree: false
fill_method: "ffill_then_zero"
ffill_limit: 5
add_missing_mask: false
winsorize:
enabled: true
by: "date"
lower_q: 0.01
upper_q: 0.99
zscore:
enabled: false
by: "date"
ddof: 0
clip: null
save_parquet: true
save_path: "{preprocess_dir}/preprocessed_{mode}*{label}*{date_start}*{date_end}.parquet"
save_state_json: true
state_path: "{model_dir}/preprocess_state_fit*{label}*{date_start}*{date_end}.json"
该模块负责数据清洗与标准化，建议 sweep 与 rolling 使用同一套预处理配置，确保不同参数组的差异只来自模型参数而非数据处理。nan_policy=tree_friendly 允许 NaN 存在，适配树模型；若你切换到线性/神经网络应改为 strict 并配合填充策略。winsorize 按 date 横截面分位裁剪去极值；zscore 当前关闭，若开启也建议按 date 做横截面标准化。save_state_json 会保存预处理状态（丢弃列、填充策略、最终特征列表），用于推理阶段保持特征一致性；这对 sweep 同样关键，因为不同参数组必须使用同一份特征列表。save_parquet 建议开启以便后续 streaming 切窗从 parquet 流式读取，减少内存压力。

datasplit:
strategy: "step_then_ratio"
window_ratio: 0.7
min_window_dates: 5
step_ratio: 0.07
train_ratio: 0.9
valid_ratio: 0.0
test_ratio: 0.1
return_dfs: false
该模块用于生成滚动窗口与窗口内切分。sweep 模式下你明确不使用 valid，因此将 valid_ratio 设为 0.0，并将 train/test 划分为 0.9/0.1。此时 evaluator.parts 也应与之保持一致（只评估 train/test），否则会出现“评估找不到 valid 文件”或指标缺失。return_dfs=false 表示 streaming 模式，不会一次性把所有窗口数据常驻内存，而是每一步按窗口读取预处理 parquet，更适合 sweep 这种重复跑多组参数的场景。

datacutting:
mode: "tree"
date_col: "date"
stockid_col: "stockid"
label_col: "y"
weight_col: null
feature_cols: null
return_dataframe_X: true
cache:
enabled: false
mode: "parquet_long"
dir: "{run_dir}/cut"
file_tpl: "{model}_{part}*fold{fold}.parquet"
meta_tpl: "{model}*{part}_fold{fold}*meta.json"
npz_tpl: "{model}*{part}*fold{fold}.npz"
keys_tpl: "{model}*{part}_fold{fold}_keys.parquet"
seq_len: 60
incomplete_policy: "drop"
pad_side: "left"
min_len_to_keep: 1
return_mask: true
mask_dtype: "bool"
strict_check_nan: false
该模块负责将 long-format 数据切成模型输入。mode=tree 表示输出二维特征矩阵用于树模型；如果未来要跑序列模型则切换为 seq 并使用 seq_len 等参数。feature_cols=null 表示自动推断特征列（排除 date/stockid/label/weight 等），因此必须保证 preprocess_state 里的最终特征列稳定，否则训练与推理会对不上。cache.enabled 建议 sweep 初期关闭避免生成大量中间文件；如果你后续发现 cut 的耗时成为瓶颈，可以开启 cache 并使用按窗口/part 的缓存文件名模板。strict_check_nan=false 在 tree 模式下通常可接受；若你切换到 strict 或 nn，建议开启以避免 silent NaN。

model:
family: "tree"
name: "lgbm"
candidates:
lgbm: {...}
xgb: {...}
catboost: {...}
该模块定义模型家族与默认参数模板。sweep 模式下 candidates 的作用主要是提供“基线默认值”，真正参与训练的参数会被 sweep.param_sets[*].model.params 覆盖。框架不强制统一参数名，各模型按原生参数名写入即可，builder 会通过 kwargs 透传。注意：如果 sweep.param_sets 里只提供部分参数，未提供的参数会从 candidates 或模型默认值继承，因此建议在 candidates 里写齐通用必要参数（objective/random_state/n_jobs 等），在 sweep 里只改动你要对比的那一小部分超参。

trainer:
name: "sweep_rolling"
params:
metric: "rankic"
maximize: true
date_col: "date"
stockid_col: "stockid"
device: "auto"
task: "regression"
loss:
enabled: true
name: "mse"
tuner:
enabled: false
update_gate: null
saver:
enabled: true
params:
dir: "{pred_dir}"
save_parts: ["train","test"]
file_tpl: "step{step_id}*fold{fold}*{part}.parquet"
model_save:
enabled: true
strategy: "last"
out_dir: "{run_dir}/sweeps/{param_id}/model"
filename: "model.pkl"
该模块是 sweep 的核心差异点。name 必须切换为 sweep_rolling，表示按 sweep.param_sets 批量运行并为每一组参数创建独立的输出目录。metric 与 maximize 用于统一比较口径，例如用 rankic 作为窗口级评分。device 建议用 auto，便于未来自动切换 CPU/GPU。loss 指定训练损失函数，sweep 下通常固定不变以专注对比模型结构参数。sweep 模式必须关闭 tuner.enabled，否则会变成“每组参数里又做寻参”，不仅浪费时间且会破坏公平对比；schedule/update_gate 也建议关闭或删除，避免不同步进触发参数更新逻辑。saver.save_parts 必须与 datasplit 的 part 集一致，你这里是 train/test 二分，所以只保存 train/test 预测文件。model_save.out_dir 建议包含 {param_id}，确保每组参数的模型不会互相覆盖；strategy 用 last 可以保证每组参数最后一个窗口的模型落盘，若你需要“按某指标挑最优窗口模型”，可扩展 strategy 但需保持 sweep 比较一致性。

sweep:
param_sets:
- id: lgbm_01
model:
family: "tree"
name: "lgbm"
params: {...}
- id: lgbm_02
...
该模块定义 sweep 要跑的参数集合列表。每个 param_set 至少需要一个唯一 id，用于生成子目录与在 sweep_summary 中标识；model.name 指定具体模型（例如 lgbm），params 填写该组要覆盖的超参数。推荐只在 params 中写“你要对比的参数”，其余保持 candidates 默认值，从而避免每组参数重复写一大坨并降低维护成本。所有 param_sets 应共享相同的数据区间、相同 preprocess 状态、相同 datasplit 切窗规则，否则对比将不再是单变量公平比较。

evaluate:
name: "default"
params:
task: "regression"
pred_dir: "{pred_dir}"
out_dir: "{eval_dir}"
save_fig: true
parts: ["train", "test"]
date_col: "date"
daily_agg: "mean"
q_bins: 10
regression:
metrics: ["rankic", "icir", "mse", "mae", "r2"]
ic_method: "spearman"
max_files: null
return_pred_df: false
stitch_regression: true
stitch_keep_individual: true
stitch_filename: "regression_all_plots_long.png"
该模块用于从每组参数的 preds 目录读取预测结果、计算指标并生成图。parts 必须与 datasplit 对齐，这里只评估 train/test。最关键的注意点是 stitch_keep_individual 必须为 true，否则 evaluator 会把 8 张单图删除，只剩一张长图，导致 sweep_compare 无法按“同类型图”做横向拼接对比。stitch_regression 可为 true 以额外产出长图，但 sweep_compare 通常更依赖单图进行同图对比，长图更多用于单次实验快速浏览。ic_method 控制 rankic 的计算方法（spearman/pearson），建议与 trainer.metric 的口径一致。

sweep_compare:
enabled: true
summary_path: "{run_dir}/sweeps/sweep_summary.parquet"
out_dir: "{eval_dir}/sweep_compare"
n_panels: 8
panel_idxs: [1,2,3,6]
sort_by: "rankic"
make_metric_bar: true
make_panel_compare: true
该模块用于 sweep 结束后的汇总对比。summary_path 指向 sweep_summary.parquet（由 sweep trainer 写出），其中记录每个 param_id 的参数、产物路径、窗口数与评估指标。panel_idxs 用于选择要横向对比的图类型索引，你当前选择的是 test quantile、mean return by quantile、rankic distribution、scatter。sort_by 用于按某个 test 指标排序展示（例如 rankic），便于把表现更好的参数组排在更靠前的位置。make_metric_bar 会生成指标柱状图对比，make_panel_compare 会把同类型图按参数组拼接成对比图。若 out_dir 为空通常意味着 evaluator 未保留单图或 sweep_summary 中记录的 eval_dir/pred_dir 路径与实际不一致，应优先检查 stitch_keep_individual 与路径占位符是否被正确渲染。

post_train_predict:
enabled: false
asof_date: "2024-06-14"
target_date: "2024-06-17"
out_path: "{pred_dir}*infer/pred*{target_date}.parquet"
该模块在 sweep 中通常关闭，因为 sweep 会跑多组参数，若开启会出现“只对最后一组参数做推理”或“推理产物路径覆盖”的歧义，且 infer 文件可能被 evaluator 误读或干扰统计。只有当你明确指定要对某一组 param_id 做推理并将 out_path 写到对应子目录时，才建议在 sweep 模式下启用。



## 4.3 预测模式（Predict）
运行方式如下：
python -m ret_pred.main --config ret_pred/config_pred.yaml
在预测模式下，系统不会执行训练流程，而是加载指定实验目录下已保存的模型 bundle，对给定日期的数据进行推理并输出预测结果。与训练模式相比，预测模式不构建标签、不执行滚动窗口、不进行调参，也不触发评估模块。
预测模式整体流程为：加载模型 → 加载当日特征数据 → 对齐特征列与训练阶段一致 → 执行推理 → 输出预测结果。

### 4.3.1 关键配置说明（config_train.yaml）
run_id 必须与训练阶段保持一致，因为模型文件将从该实验目录的 model_dir 中加载。如果 run_id 不匹配，将无法找到对应模型。
task 模块中，mode 必须设置为 "predict"，同时 build_label 必须设为 false，因为推理阶段不需要真实标签。虽然仍然保留 label_col 字段以保持数据结构一致，但预测阶段不会使用该列进行训练或评估。

dataloader部分与训练阶段基本一致，但通常只加载单日数据，例如：
date_start: "2024-06-17"
date_end:   "2024-06-17"

preprocess预处理模块仍然会执行与训练阶段一致的缺失值策略与特征处理逻辑，以确保特征工程完全一致，但通常会关闭样本与列删除（例如将 drop 阈值设为大于 1），避免在推理阶段意外丢失特征。

model模型模块只需要指定 family 和 name，实际模型参数不会重新初始化，而是通过 bundle_dir 加载训练阶段保存的模型。

预测核心配置如下：
predict:
  asof_date: "2024-06-17"
  target_date: "2024-06-18"
  bundle_dir: "{model_dir}"
  out_path: "{pred_infer_dir}/pred_{target_date}.parquet"

其中 asof_date 表示特征数据对应的日期，target_date 表示预测目标日期。系统会加载 bundle_dir 中保存的模型，对 asof_date 的特征进行推理，并将预测结果保存为指定路径的 parquet 文件。

# 5. 实验目录结构与输出产物说明
每一次训练或预测都会根据 run_id 自动生成独立实验目录。以当前实验 exp001 为例，完整目录结构如下：
runs/
  exp001/
    logs/
      train.log
    preprocess/
      preprocessed_fit_y_2024-06-01_2024-06-14.parquet
    model/
      model.pkl
      meta.json
      feature_cols.json
      preprocess_state_fit_y_2024-06-01_2024-06-14.json
    preds/
      step0_fold0_train.parquet
      step0_fold0_valid.parquet
      step0_fold0_test.parquet
      step1_fold1_train.parquet
      ...
    eval/
      metrics.csv
      regression_all_plots_long.png
    preds_infer/
      pred_2024-06-17.parquet

下面对各目录与文件进行详细说明。
logs/
  train.log
该目录用于保存运行日志文件。日志内容包括数据规模、预处理统计信息、滚动窗口信息、调参过程、模型保存信息以及预测阶段输出信息。日志系统支持文件自动轮转，当日志文件达到设定大小后会自动生成新的日志文件，确保长期实验记录不会被覆盖。

preprocess/
  preprocessed_fit_y_2024-06-01_2024-06-14.parquet
该目录保存预处理后的 long-format 数据文件。文件名中包含运行模式（fit 或 predict）、标签名称以及日期区间。训练阶段所有滚动窗口数据均基于该 parquet 文件进行读取。此文件已经完成缺失值处理、去极值、标准化等特征工程步骤，是模型训练的直接输入数据。

model/
  model.pkl
  meta.json
  feature_cols.json
  preprocess_state_fit_y_2024-06-01_2024-06-14.json
该目录保存最终选定模型及相关元信息。
model.pkl 为最终保存的模型文件，是预测阶段加载的核心对象。
meta.json 保存模型选择与训练窗口的元信息，包括所使用模型类型、最终参数、参数哈希值、对应 step 与 fold 编号、验证集得分以及模型保存时间等信息 
feature_cols.json 保存训练阶段最终使用的特征列列表，用于预测阶段进行特征对齐，确保训练与推理使用完全一致的特征集合 
preprocess_state_fit_*.json 保存预处理阶段的详细状态信息，包括样本数量变化、删除的标签行数、各特征缺失比例、winsorize 配置以及最终特征列表等 

preds/
  step{step_id}_fold{fold}_{part}.parquet
该目录保存滚动训练过程中生成的预测结果文件。每个文件对应一个时间窗口中的一个数据部分，其中 part 可能为 train、valid 或 test。文件内容通常包括预测值、真实值、日期与股票代码等字段，用于评估、回测或外部策略调用。
由于使用滚动窗口训练，目录中可能包含多个 step 与 fold 组合的预测文件。

eval/
  metrics.csv
  regression_all_plots_long.png
该目录保存评估阶段生成的指标与图像结果。
metrics.csv 为实验指标汇总文件，包含 train、valid、test 各阶段的回归指标，例如 RankIC、ICIR、MSE、MAE、R2 等 。
regression_all_plots_long.png 为评估模块生成的综合可视化图像，包含分位累计收益图、分位均值收益曲线、RankIC 分布、残差分布、真实值与预测值散点图以及日度聚合曲线等，用于全面评估模型排序能力与拟合稳定性。

preds_infer/
  pred_2024-06-17.parquet
当配置中启用 post_train_predict 时，系统会在训练完成后自动执行一次推理，并将预测结果保存至该目录。该文件通常仅包含预测值、日期与股票代码，不包含真实标签，适用于实盘前信号生成或下游策略系统调用。


# 6. 扩展方式
新增模型
在 tree_models/ 或 nn_models/ 添加模型文件
在 registry 注册
在 config 中指定 name

新增损失函数
在 losses/ 添加实现
在 registry 注册
config 中指定 loss.name

新增调参策略
在 trainer/plugins.py 中实现新的 tuner 即可。


# 7. 数据流梳理
============================================================
STEP 1) dataloader.load_long
------------------------------------------------------------
输入来源：
    原始 parquet 文件（全区间）

输入格式（示例）：
    parquet long-format 表：
        date        stockid     f1     f2     ...   y(optional)
        2020-01-02  000001.SZ   0.12   1.03         0.005
        2020-01-02  000002.SZ   -0.34  0.56        -0.002
        ...

输出：
    long_df : pd.DataFrame（long-format，全区间）
        列结构：
            [date, stockid, (optional y), features...]

    meta : dict
        记录字段来源、label来源等信息

------------------------------------------------------------
STEP 2) preprocess.preprocess_long（fit）
------------------------------------------------------------
输入来源：
    long_df（来自 dataloader）

输入格式：
    pd.DataFrame（long-format，全区间）
        [date, stockid, y, features...]

输出：
    clean_df : pd.DataFrame（long-format，全区间）
        列顺序稳定：
            [date, stockid, (optional y),
             base_feature_cols..., mask_cols...]

        示例：
            date        stockid    y     f1     f2    f1__miss
            2020-01-02  000001.SZ  0.005  0.23  -0.45   0
            2020-01-02  000002.SZ -0.002 -0.11   0.78   1

    pp_state : dict
        - base_feature_cols
        - mask_cols
        - 标准化统计量
        - fill统计量
        - drop信息

若 save_parquet=True：
    clean_df 会落为“全区间 parquet 文件”


------------------------------------------------------------
STEP 3) split.datasplit_long
------------------------------------------------------------
输入来源：
    clean_df[[date_col]]

输入格式：
    仅包含一列 date 的 DataFrame

        date
        2020-01-02
        2020-01-03
        ...

输出：
    folds : List[dict]

    每个 fold 示例：
        {
            "train_dates": [...],
            "valid_dates": [...],
            "test_dates": [...],
            "window_start": "2020-01-02",
            "window_end": "2020-06-30",
            "fold": 0
        }


------------------------------------------------------------
STEP 4) windows.build_streaming_windows
------------------------------------------------------------
输入来源：
    - preprocess 落的“全区间 clean parquet”
    - folds（日期列表）

输入格式：
    clean parquet（long-format，全区间）

行为：
    对每个 fold：
        按 train_dates / valid_dates / test_dates
        从 parquet 中筛选子集

输出（iterator，每次 yield）：
    train_df, valid_df, test_df, meta0

    示例：
        train_df（long-format）:
            date        stockid    y     f1    f2 ...
            2020-01-02  000001.SZ  0.005 ...
            ...

        meta0:
            {
                "fold": 0,
                "window_start": "...",
                "window_end": "..."
            }

------------------------------------------------------------
STEP 5) cut.datacut_long（tree）
------------------------------------------------------------
输入来源：
    train_df / valid_df / test_df（来自 windows）

输入格式：
    pd.DataFrame（long-format，单窗口）

        date    stockid    y    f1   f2   ...
        ...

输出：
    payload : dict

        {
            "X": np.ndarray 或 DataFrame
                 形状: (n_samples, n_features)

            "y": np.ndarray
                 形状: (n_samples,)

            "keys": pd.DataFrame
                 列: [date, stockid]

            "feature_cols": List[str]
        }

    示例：
        X.shape = (120000, 35)
        y.shape = (120000,)
        keys.shape = (120000, 2)

    state : dict
        记录 n_samples / n_features 等信息


------------------------------------------------------------
STEP 6) trainer.RollingTrainer.run
------------------------------------------------------------
输入来源：
    windows 迭代器输出的 payload

输入格式：
    train_pl / valid_pl / test_pl
        含 X / y / keys / feature_cols

行为：
    逐窗口训练模型
    生成预测
    计算指标
    保存 pred 文件

输出：
    history_df（训练记录表）

    pred 文件（parquet）
        示例：
            date    stockid    y_true    y_pred    part
            2020-06-30 000001.SZ  0.005   0.0042   test


------------------------------------------------------------
STEP 7) evaluate.Evaluator.run
------------------------------------------------------------
输入来源：
    pred_dir 中保存的预测 parquet 文件

输入格式：
    多个 pred_*.parquet

输出：
    指标结果
    图表
    汇总报告
"""