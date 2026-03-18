# ret_pred 配置设计（schema 草案）

本文档用于把 `ret_pred/md/blueprint.md`、`ret_pred/md/data.md`、`ret_pred/md/tasks.md` 中提到的新增能力，
落到可执行的 YAML 配置结构上，并明确与旧配置的兼容策略。

## 1. 兼容策略（必须）

- 继续沿用现有入口与分派方式：`trainer.name` 决定训练器类型。
  - 已有：`rolling`、`sweep_rolling`
  - 新增（后续实现）：`rankic_refit_roll`
- 所有新增功能默认 `enabled: false`，以保证旧工作流行为不变。
- 新增字段只做“可被读取”的 schema 落地；在对应功能未实现前不会生效。

## 2. 顶层结构概览

典型训练配置结构（关键段落）：

- `task`
- `logging`
- `paths`
- `dataloader`
- `feature_engineering`（新增，默认关闭）
- `preprocess`（扩展 zscore.method）
- `datasplit`
- `datacutting`
- `model`
- `trainer`（新增 trainer.name=rankic_refit_roll）
- `evaluate` / `predict`（可选，按现有逻辑；evaluate 新增 future_concat 支持）

## 3. dataloader（新增 label_postprocess）

落点：`dataloader.label_postprocess`

语义（见 `data.md`）：
- raw label 构造完成后，对 `y` 按交易日做横截面后处理；
- 默认顺序固定：**先 zscore 再 winsorize**；
- 不修改原始价格列；
- 通过配置显式启用，默认关闭。

建议字段：
```yaml
dataloader:
  label_postprocess:
    enabled: false
    keep_y_raw: false

    zscore_by_date:
      enabled: true
      ddof: 0
      eps: 1.0e-12

    winsorize_by_date:
      enabled: true
      lower_q: 0.05
      upper_q: 0.95
```

重要提醒：
- 若启用 `keep_y_raw`，必须确保 `y_raw` 不会被自动推断为 feature 列（避免泄露/口径错误）。

## 4. feature_engineering（新增模块）

落点：顶层 `feature_engineering`

语义：
- 位置固定在 `dataloader -> feature_engineering -> preprocess`；
- 按 `stockid` 分组、按 `date` 排序，只用历史，不得用未来信息；
- 输出仍为 long-format，新增列追加回 DataFrame。

建议字段：
```yaml
feature_engineering:
  enabled: false
  rolling_stats:
    enabled: true
    windows: [5, 10, 20]
    stats: ["mean", "std"]
    min_periods: 1
```

## 5. preprocess（扩展 robust zscore）

落点：`preprocess.zscore.method`

语义（见 `data.md` / `tasks.md`）：
- `method: standard` 时保持旧行为；
- `method: robust` 时按（date 截面或 global）使用 `median + MAD`：
  - `z = (x - median) / (MAD + eps)`
  - 可选乘 `0.6745`（通过 config 控制）。

建议字段：
```yaml
preprocess:
  zscore:
    enabled: false
    by: "date"          # date | global
    method: "standard"  # standard | robust
    ddof: 0
    clip: null
    robust:
      eps: 1.0e-12
      use_06745: false
```

## 6. trainer（新增 rankic_refit_roll：仅 schema，后续实现）

落点：`trainer.name: rankic_refit_roll`

语义要点（见 `tasks.md` / `data.md`）：
- 必须严格区分：
  - **test evaluation**（eval model 的 test 指标与 test preds）
  - **future prediction**（final model 预测 test 之后未来 `N` 天，并拼接用于统一回测）
- 因子集合：
  - 每轮重训节点可更新；
  - 同一轮内从训练到 future prediction 必须固定，不允许按天更换。

建议字段（占位，细节以实现阶段确认为准）：
```yaml
trainer:
  name: "rankic_refit_roll"
  params:
    windowing:
      mode: "days"   # days | ratio

      # --- mode=days（纯数字，按交易日个数理解） ---
      initial_train_days: 252
      eval_test_days: 20
      prediction_horizon_days: 5
      step_days: null          # null => step=prediction_horizon_days（future block 不重叠）

      # --- mode=ratio（按全样本交易日数换算） ---
      # initial_train_ratio: 0.7
      # eval_test_ratio: 0.1
      # prediction_horizon_ratio: 0.02
      # step_ratio: null
      # min_initial_train_days: 60
      # min_eval_test_days: 5
      # min_prediction_horizon_days: 1

      expanding_train: true    # true: 从起点扩展到 train_end；false: 固定窗口（需 max_train_days）
      max_train_days: null

    factor_selection:
      enabled: false
      predictions_pkl_path: "/path/to/predictions.pkl"
      rankic_threshold: 0.03
      aggregate: "abs_mean"          # abs_mean | mean_abs
      cutoff_inclusive: true         # true => 允许使用 <= asof_date 的记录（用户确认）
      min_history_days: 1
```

不确定性声明（需要在实现阶段向用户确认）：
- `predictions.pkl` 的具体字段结构与“历史可见性 cutoff（lag）”约束目前只在语义层描述，
  实现前需要确认该文件的真实格式与可用日期边界，避免未来信息泄露。

窗口语义说明：
- 每轮 window 定义包含：train、test evaluation、future block 三段，且 future block 长度为 N。
- `factor_selection_asof_date` 固定使用该轮 `train_end`（用户确认），并按 `cutoff_inclusive` 决定是否包含当天记录。
- 尾部若不足一个完整 future block：直接停止（不输出不完整尾块，便于 future 拼接回测语义清晰）。

## 7. evaluate（适配 rankic_refit_roll 的 future concat backtest）

对于 `rankic_refit_roll`，需要严格区分：
- `part=test`：test evaluation（用于正式 OOS 评估）
- `part=future`：future prediction（用于拼接形成连续样本外序列并统一回测）

Evaluator 新增可选配置（默认关闭，保持旧行为）：
```yaml
evaluate:
  params:
    parts: ["test", "future"]   # 至少包含 future 才会被加载
    stockid_col: "stockid"
    future_concat:
      enabled: true
      future_part: "future"     # 与 saver 输出一致
      out_subdir: "future"      # 输出到 {eval_dir}/future
      preds_filename: "future_preds_all.parquet"
      validate_no_overlap: true # 检查 (date,stockid) 是否重复（future block 不应重叠）
      save_fig: true            # 是否为 stitched future 序列单独画图
```

输出产物（在 `{eval_dir}`）：
- `metrics_test.csv`：仅 test evaluation 指标
- `{eval_dir}/future/future_preds_all.parquet`：future 序列拼接结果
- `{eval_dir}/future/metrics_future.csv`：基于 stitched future 序列的指标

## 8. evaluate（接入已有因子评测系统，功能块 G）

背景（见 `tasks.md` 功能块 G / `blueprint.md` 10.7 / `data.md` 14A）：
- 框架需要复用已有的因子评测系统（`run_pred_factor_eval.py` / `eval_all.pyc`），但这些文件为只读依赖，禁止修改；
- `rankic_refit_roll` 必须用拼接后的 `future_preds` 作为因子评测输入（不能用 test evaluation preds）；
- 其他训练策略使用其常规 prediction 拼接结果进入同一套系统；
- 评测必须支持 `excess_return=True/False/both` 三种输出模式，并且双输出时目录必须隔离。

落点：`evaluate.params.factor_eval`

建议字段：
```yaml
evaluate:
  params:
    # ... 原有 evaluator 参数

    # 可选：接入已有因子评测系统（默认关闭；仅服务器环境可跑）
    factor_eval:
      enabled: false

      # 必填：训练策略名（用于决定默认输入口径）
      strategy: "rankic_refit_roll"       # rankic_refit_roll | rolling | sweep_rolling | ...

      # 可选：评测输入来源（默认 auto）
      # - auto:
      #     - rankic_refit_roll => future_stitched（拼接后的 future 序列）
      #     - others            => stitched_part（默认 part_name=test）
      # - future_stitched: 使用 {eval_dir}/future/future_preds_all.parquet（或 future_parquet_path 覆盖）
      # - stitched_part  : 从 pred_dir 读取 parquet，按 part_name 过滤后 stitch
      # - explicit_parquet: 直接读 future_parquet_path（用户自备 stitched 文件）
      input_source: "auto"                # auto | future_stitched | stitched_part | explicit_parquet
      part_name: "test"                   # input_source=stitched_part 时生效
      pred_glob: "**/*.parquet"           # 可选：用于 stitched_part 扫描
      future_parquet_path: null           # 可选：覆盖 stitched future parquet 路径

      # 可选：y_pred 列名（默认 y_pred）
      value_col: "y_pred"

      # 必填：输出目录（支持占位符；推荐显式包含 strategy/source/excess）
      # 可用占位符：{eval_dir} {strategy} {input_source} {source_tag} {excess_mode}
      out_dir: "{eval_dir}/factor_eval_{strategy}_{source_tag}_{excess_mode}"

      # 可选：用于 out_dir 命名的 source tag（默认按策略推断）
      # - rankic_refit_roll + (auto/future_stitched) => future_preds
      # - else => {part_name}_preds
      source_tag: null

      # 可选：excess_return 输出模式
      excess_mode: "both"                 # true | false | both

      # 可选：excess 输出目录布局（默认 subdirs，保持兼容；separate_dirs 更贴近 tasks.md 示例）
      # - subdirs       : out_dir/excess_true/ 与 out_dir/excess_false/
      # - separate_dirs : out_dir_excess_true/ 与 out_dir_excess_false/
      excess_output_layout: "subdirs"     # subdirs | separate_dirs

      # 可选：评测参数回显（meta）文件名（每个输出目录都会写一份）
      meta_filename: "factor_eval_meta.json"

      # 必填：complete_factor_analysis 所需参数（全部从 config 给出；不要依赖默认值）
      analysis_params:
        factor_name: "my_factor"
        save_figures: true
        return_data: false
        verbose: true
        # 其余字段按已有系统要求补全，例如：
        # start_date: "2024-01-01"
        # end_date: "2024-06-14"
        # ...（在服务器环境对齐 run_pred_factor_eval.py 需要的口径）
```

重要提醒：
- `rankic_refit_roll` 若使用 `input_source=future_stitched/auto`，需要先产出 stitched future 文件（通常依赖 `future_concat.enabled=true`）。
- 因子评测系统依赖 clickhouse 连接与其 json 配置，通常只在服务器可运行；本框架只负责桥接与调度。
