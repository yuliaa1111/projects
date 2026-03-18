# factor eval runbook（功能块 G：最小验证步骤）

本文用于指导在**服务器环境**最小化验证“接入已有因子评测系统”是否跑通。

> 说明：已有因子评测系统依赖 clickhouse 连接与其 json 配置路径（由 `ret_pred/run_pred_factor_eval.py` 负责寻找并连接）。
> 本仓库只做桥接与调度，不修改任何只读依赖文件。

## 1. 前置条件

- 已有因子评测系统相关文件存在且可导入（只读依赖）：
  - `ret_pred/run_pred_factor_eval.py`
  - `ret_pred/manage_db_read.pyc`
  - `ret_pred/db_settings.json`
  - `ret_pred/eval_all.pyc`
- 服务器上 `run_pred_factor_eval.py` 能找到其 clickhouse 配置 json（路径规则以该文件内部为准）。
- 训练或预测流程已生成预测 parquet：
  - `rankic_refit_roll`：必须包含 `part=future`，并通过 `future_concat` 产出 stitched 文件：
    - `{eval_dir}/future/future_preds_all.parquet`
  - 其他策略：至少有 `part=test` 的预测 parquet（或你在 config 指定的 part）。

## 2. 推荐的 config 最小配置（rankic_refit_roll）

关键段落（示例见 `ret_pred/config_train_rankic_refit_roll_factor_eval.yaml`）：

- `evaluate.params.future_concat.enabled: true`
- `evaluate.params.factor_eval.enabled: true`
- `evaluate.params.factor_eval.strategy: "rankic_refit_roll"`
- `evaluate.params.factor_eval.input_source: "auto"`（默认会走 stitched future）
- `evaluate.params.factor_eval.excess_mode: "both"`（或 true/false）
- `evaluate.params.factor_eval.analysis_params: {...}`（必须与服务器侧 `complete_factor_analysis` 签名对齐）

## 3. 最小运行命令

在服务器上运行（示例）：

```bash
python3 -m ret_pred.main --config ret_pred/config_train_rankic_refit_roll_factor_eval.yaml
```

或你自己的训练 config（确保打开上述开关）。

## 4. 验证点（必须）

### 4.1 口径正确性（避免混用 test evaluation）

- 对 `rankic_refit_roll`：
  - 因子评测输入必须来自 stitched future（默认路径）：
    - `{eval_dir}/future/future_preds_all.parquet`
  - 不得使用 `part=test` 的评估预测。

### 4.2 产物命名与不覆盖

当 `excess_mode: both`：

- 若 `excess_output_layout: separate_dirs`：
  - `{eval_dir}/factor_eval_rankic_refit_roll_future_preds_excess_true/`
  - `{eval_dir}/factor_eval_rankic_refit_roll_future_preds_excess_false/`
- 若 `excess_output_layout: subdirs`：
  - `{eval_dir}/factor_eval_rankic_refit_roll_future_preds/excess_true/`
  - `{eval_dir}/factor_eval_rankic_refit_roll_future_preds/excess_false/`

并且：
- 每个输出目录内必须存在 `factor_eval_meta.json`（或你配置的 `meta_filename`）。

### 4.3 meta 回显检查

打开 `factor_eval_meta.json`，重点检查字段：
- `resolved_input_source`：应为 `future_stitched`（或你显式指定的来源）
- `excess_return`：分别为 true/false
- `analysis_params`：与 config 一致（用于复盘）

## 5. 常见问题排查

- `future stitched parquet not found`：
  - 说明没有先产出 `{eval_dir}/future/future_preds_all.parquet`；
  - 检查 `evaluate.params.future_concat.enabled=true` 且 `parts` 中包含 `future`。
- clickhouse 连接失败：
  - 由 `run_pred_factor_eval.py` 的配置路径与服务器环境决定；
  - 本框架不负责修复连接逻辑（且禁止修改只读依赖文件）。
- `analysis_params` 缺字段/签名不匹配：
  - 必须在 config 里补齐 `complete_factor_analysis` 需要的全部参数；
  - 本框架不会给隐式默认值（符合任务要求“全部从 config 给出”）。

