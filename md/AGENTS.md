# AGENTS.md (repo root)

本文件定义本仓库（`ret_pred/`）的开发规则与协作约束，面向 AI agent / Codex 与人类开发者。
内容基于：`ret_pred/md/blueprint.md`、`ret_pred/md/data.md`、`ret_pred/md/tasks.md`。

## 0. 必读文档（改代码前）
- `ret_pred/md/blueprint.md`：目标架构、重构原则、模块职责与新增功能清单。
- `ret_pred/md/data.md`：数据契约、label/feature/preprocess/rolling 的语义与硬约束。
- `ret_pred/md/tasks.md`：功能块拆解、验收标准、推荐开发顺序与回归验证点。

## 1. 总体开发原则（硬约束）
- **增量式改造**：优先最小可运行改动，分功能块逐步合并；避免“大重写”。
- **配置驱动**：新增能力必须可通过 YAML 配置开关/参数控制；默认关闭以保证旧工作流不变（除非明确要求默认开启）。
- **数据语义不可悄改**：不得静默改变 label 定义、评估口径、切分口径、winsorize/zscore 顺序等。
- **严格避免未来信息泄露**：
  - 所有 rolling/rolling feature 必须只用当前及历史数据；
  - 动态选因子必须只使用截至重训节点 `t` 可见的历史信息；
  - 同一轮（一个 refit 周期）内的因子集合必须固定，不能按天更换。
- **训练/推理解耦**：训练期 `fit` 统计与推理期 `transform` 复用 state 的边界必须清晰。
- **产物可复盘**：关键中间结果与每轮选择（如 selected_factors、eval/final preds）要可追踪、命名清晰。

## 2. 规范数据契约（必须遵守）
- 内部标准数据结构：**long-format DataFrame**，一行一个 `(date, stockid)` 样本。
- 关键列：`date_col`（默认 `date`）、`stockid_col`（默认 `stockid`）、`label_col`（通常 `y`）。
- `(date, stockid)` 理想情况下唯一；重复键必须显式处理或直接报错（避免训练/评估异常）。

## 3. 目标主流程（模块边界）
目标链路（新增功能落地后）应保持顺序清晰：
`dataloader` →（可选）`label_postprocess` →（可选）`feature_engineering` → `preprocess` → `split/windows` → `trainer` → `evaluate` →（可选）`predictor`

### 3.1 dataloader（数据输入层）
- 只负责：读数据（long/wide）、feature_mask 字段筛选、构造 raw label（如基于价格列 shift 的 forward return）。
- **label_postprocess（新增）**必须放在 dataloader 内部（raw label 构造之后）：
  - 按日横截面 zscore → 按日横截面 winsorize（顺序不可变）；
  - 不修改原始价格列；
  - 可选保留原始 `y` 为 `y_raw`（但必须避免进入特征集合，见“禁止事项”）。

### 3.2 feature_engineering（特征增强层，新增模块）
- 只负责：对原始 feature 列做时间统计特征（rolling mean/std 等），按 `stockid` 分组、按 `date` 排序、不得用未来信息。
- 输出仍为 long-format，新增列追加回原 DataFrame。

### 3.3 preprocess（预处理层）
- 只负责：清洗（date/inf）、缺失策略、winsorize、zscore（含 robust method）、state 管理（fit/transform）。
- robust zscore 必须作为 zscore 的 `method` 分支实现，而不是复制一套新 preprocess 流程。

### 3.4 split/windows（时间切分层）
- 切分必须**按日期切**（不是按行随机切），保证时间序列语义与避免泄露。
- windows 负责从 preprocess 落盘产物中“流式”读取窗口数据并切出 train/valid/test（或新策略需要的其它分段）。

### 3.5 trainer（训练策略层）
- 旧策略必须保持可用（如 rolling / sweep_rolling）。
- 新策略 `rankic_refit_roll` 的语义必须严格区分：
  - **test evaluation**：eval model 在 test 上的正式样本外评估（输出 test 指标与 test preds）；
  - **future prediction**：final model（train+test 重训）预测 test 后未来 `N` 天（future block preds 用于拼接统一回测）；
  - 两者不得混用、不得回填、不得直接拼 test preds 做最终回测。

### 3.6 factor_selection（因子筛选层，新增逻辑）
- 读取 `predictions.pkl`，在重训节点 `t` 只能使用截至 `t` 的历史信息计算/聚合因子有效性并筛选。
- 每轮选出的因子集合必须落盘为 artifact，供复盘。

### 3.7 evaluate（评估层）
- 必须能区分并分别输出：
  - test 阶段指标/图；
  - future preds 拼接后的统一回测指标/图。

### 3.8 predictor（推理层）
- 推理必须复用训练期 preprocess state，并确保特征对齐。
- 若训练引入 rolling 特征，推理侧必须有清晰策略生成相同特征（避免大量缺列补 NaN 导致结果失真）。

## 4. 禁止事项（违规即回滚）
- 禁止静默改变以下内容（除非用户明确要求并写入文档/配置变更说明）：
  - label 定义与计算口径（尤其是 forward return 的时间对齐）；
  - winsorize/zscore 的默认顺序或计算口径；
  - datasplit 以日期为单位切分的语义；
  - evaluator 的指标定义与口径；
  - 训练/推理 state 复用机制（fit/transform 边界）。
- 禁止引入未来信息泄露：
  - rolling 特征不得使用未来；
  - 动态选因子不得读取 `t+1` 及之后的信息；
  - 同一轮（一个 window）内不得按天更换因子集合。
- 禁止把 `y_raw`、`y` 或任何“未来收益信息”误当作特征列输入模型。
- 禁止把“业务逻辑”混进错误模块（例如把 label postprocess 写进 preprocess，或把 feature_engineering 写进 dataloader 的读数逻辑里）。
- 禁止为了“更干净”删除现有可用工作流（rolling/sweep/predict/evaluate）或破坏产物目录结构。

## 5. 每次改动后的汇报格式（提交 PR/发消息时照此输出）
1. **改了哪些文件**：逐文件列出路径。
2. **每个文件的改动目的**：一句话说明职责与接口变化。
3. **新增/变更的 config 字段**：字段路径、默认值、启用条件、与旧行为兼容性。
4. **数据流说明**：本次改动插入到 pipeline 的哪个位置，输入/输出契约是什么。
5. **未来信息泄露防护**：解释边界（按 date/stockid、按重训节点截断、同轮固定因子等）。
6. **产物与命名**：新增/调整的 artifact 文件名与目录位置（尤其是 window 级别与 all 拼接级别）。
7. **回归影响**：是否 breaking change；关闭开关时是否与旧行为一致。
8. **最小运行/验证方式**：给出 1–3 条命令或配置片段，说明如何跑通与验证关键验收点。

硬约束：run_pred_factor_eval.py、manage_db_read.pyc、db_settings.json、eval_all.pyc 只允许读取，不允许做任何修改；如果需要适配，请在当前框架中新增桥接层，不要改这些文件。