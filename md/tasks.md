# ret_pred 本次重构与功能扩展任务单（详细版）

## 1. 总体目标

本次工作不是单纯整理代码，而是要在现有框架基础上同时完成以下几类修改：

1. 保持整体数据流与量化研究语义基本不变
2. 提升模块边界清晰度与可维护性
3. 增加新的标签后处理能力
4. 增加新的特征工程能力
5. 增加新的稳健标准化能力
6. 增加新的训练策略
7. 在新训练策略中接入基于 `predictions.pkl` 的历史 RankIC 动态选因子
8. 支持按未来 `N` 天为一个区间做滚动预测，并把所有区间拼接后统一回测
9. 保证旧工作流尽量不被破坏
10. 保证所有新增能力都能通过 config 可插拔控制

Agent 在执行任务时，应优先保证：增量式改造、可运行、可验证、无未来信息泄露。

---

## 2. 任务拆解总览

本次任务拆为六个功能块：

### 功能块 A：dataloader 增加 label postprocess
### 功能块 B：新增 feature_engineering 模块
### 功能块 C：preprocess 增加 robust zscore
### 功能块 D：新增训练策略 `rankic_refit_roll`
### 功能块 E：接入基于 `predictions.pkl` 的动态选因子
### 功能块 F：main / evaluator / config / artifact 统一适配

---

## 3. 功能块 A：dataloader 增加 label postprocess

### A1. 目标
在 `dataloader.py` 中，在 label 构造完成之后，增加可选的 label 后处理流程。

### A2. 第一版要求
支持如下处理链：
1. 按交易日对 `y` 做横截面 zscore
2. 按交易日对 `y` 做横截面 winsorize

默认 winsorize 参数：
- lower_q = 0.05
- upper_q = 0.95

### A3. 具体实现要求
- 在 `build_label` 或原有 label 构造流程完成后接入
- 不要修改原始价格列
- 不要把该逻辑混入 preprocess 模块
- 该逻辑应明确归类为 `label_postprocess`
- 支持按 config 决定是否启用
- 支持未来扩展更多 label postprocess steps
- 如配置要求，可保留原始 `y` 为 `y_raw`

### A4. 推荐实现方式
建议新增一个内部函数，例如：
- `apply_label_postprocess(df, config, date_col, label_col)`

内部按步骤执行：
- zscore by date
- winsorize by date

### A5. 数据语义要求
- 默认顺序必须是先 zscore，再 winsorize
- 不得悄悄调整顺序
- 最终训练使用的 `y` 应为 postprocess 后的结果
- 关闭开关时，与旧版本行为一致

### A6. 验收标准
- 关闭开关时，与旧版本行为一致
- 开启后，`y` 会按日进行横截面 zscore + winsorize
- 不影响原始 feature 列逻辑
- 可通过 config 调整分位数参数

---

## 4. 功能块 B：新增 feature_engineering 模块

### B1. 目标
新增一个独立模块，例如 `feature_engineering.py`，用于对原始因子列构造时间统计特征。

### B2. 第一版要求
必须支持：
- rolling mean
- rolling std

默认窗口：
- 5
- 10
- 20

### B3. 具体实现要求
- 输入是 long-format DataFrame
- 按 `stockid` 分组
- 按 `date` 排序
- 对选定的原始 feature 列生成 rolling 特征
- 新特征追加回 DataFrame
- 不得使用未来信息
- 应支持 config 控制是否启用
- 输出应仍然保持 long-format

### B4. 推荐接口
建议暴露一个主函数，例如：
- `run_feature_engineering(df, config, date_col, stockid_col, feature_cols)`

内部至少可支持：
- `add_rolling_stats(...)`

### B5. 特征命名要求
建议统一命名为：
- `{feature}__roll_mean_{window}`
- `{feature}__roll_std_{window}`

### B6. 与现有流程接入位置
新模块应放在：
`dataloader -> feature_engineering -> preprocess`

### B7. 验收标准
- 关闭开关时，与旧版本行为一致
- 开启后，指定特征会新增 rolling mean / std 列
- 新增特征列数与窗口、统计量设置一致
- 不出现未来信息泄露
- 对 tree mode 和 seq mode 都能进入下游流程

---

## 5. 功能块 C：preprocess 增加 robust zscore

### C1. 目标
在 `preprocess.py` 中扩展 zscore 功能，使其支持 `robust` 模式。

### C2. 第一版要求
robust zscore 基于：
- median
- MAD

典型公式：
`z = (x - median) / (MAD + eps)`

可选支持：
- 是否乘 `0.6745`，由 config 控制

### C3. 具体实现要求
- 不要另起一套完全重复的 preprocess 流程
- 应作为 zscore 的一个 method，例如：
  - `method: standard`
  - `method: robust`
- 支持按 `date` 做截面计算
- 保持与现有 fit/transform 结构兼容
- 与 clip、winsorize 等流程的相对顺序应清晰

### C4. 推荐实现方式
建议在现有 zscore 相关函数中扩展 method 分支，而不是复制一个 `robust_preprocess.py`

### C5. 验收标准
- `method: standard` 时，与旧行为一致
- `method: robust` 时，按 date 使用 median + MAD 做标准化
- 支持通过 config 设置 `eps`
- 如配置开启 scale factor，则可乘 `0.6745`

---

## 6. 功能块 D：新增训练策略 `rankic_refit_roll`

### D1. 目标
新增一种滚动训练策略，命名为：

`rankic_refit_roll`

该名字的含义是：
- `rankic`：该策略依赖历史 RankIC 做因子筛选
- `refit`：每轮在正式 test 评估后，会在扩展样本上重新训练 final model
- `roll`：整体以滚动方式向前推进

### D2. 该策略的总体语义
每一轮并不是简单地“train 一次，test 一次”。
而是包含四个阶段：
1. 历史选因子
2. train→test 样本外评估
3. train+test 重训 final model
4. 预测 test 之后的 future block

### D3. 第一轮示例语义
例如：
- 用第 1 到 252 天作为训练样本
- 在训练开始前，根据截至该时点可获得的历史因子表现，从 `predictions.pkl` 中筛出 abs(RankIC) >= 0.03 的因子
- 得到第一轮固定使用的因子集合
- 用这套固定因子训练起始模型
- 在后续一段 test 区间上做静态样本外评估
- 在这段评估期间不再更换因子

### D4. 后续轮次语义
例如已经走完第 272 天，准备预测第 273 天时：
- 先基于截至第 272 天的历史信息重新筛一次因子
- 得到一套新的有效因子集合
- 将训练样本扩展到第 1 到 272 天
- 在这整段扩展样本上统一使用这套新筛出的因子重新拟合一个 final model
- 再去预测第 273 天，或者更一般地预测未来 N 天

### D5. 轮次内与轮次间因子集合的约束
这是整个新策略最重要的约束之一：

- 因子集合允许在不同重训轮次之间更新
- 但同一个模型一旦开始训练，从训练到对应预测都保持同一套因子
- 不允许在同一轮 test 区间或 future prediction 区间内按天更换因子

也就是说：
- 不同轮次之间可以换因子
- 同一轮内部不能换因子

### D6. test evaluation 与 future prediction 的语义区分
必须明确区分：

#### test evaluation
- 用 train split 训练的 eval model
- 在 test split 上做正式样本外评估
- 输出 test preds 和正式 test metrics

#### future prediction
- 在 train+test 上重新拟合的 final model
- 用于预测 test 之后的未来 `N` 天
- 输出 future block preds
- 这些结果最终按时间拼接用于统一回测

### D7. 为什么不能混
- test evaluation 用来评估模型在该轮的样本外泛化能力
- future prediction 用来构造最终连续的样本外预测序列
- 不能把 refit 后 final model 的结果回填到 test 指标里
- 不能把 test preds 和 future preds 混在一起直接回测

### D8. 训练器实现要求
可以：
- 新写一个 trainer 文件，例如 `trainer_rankic_refit_roll.py`
- 或在现有 trainer 框架里新增一个 strategy 分支
- 其他模块例如split,window也可以新建相应适配
但不建议把所有逻辑直接塞进一个超长函数里。

### D9. 推荐中间对象 / 产物
建议在概念上区分：
- `selected_factors`
- `eval_model`
- `eval_preds`
- `eval_metrics`
- `final_model`
- `future_preds`

### D10. 未来预测区间长度
新策略必须支持配置：
- `prediction_horizon_days = N`

语义为：
- 每轮重训后连续预测未来 `N` 天
- 下一轮的重训节点向前滚动 `N` 天
- 各轮预测区间首尾衔接，不重叠
- 最终可以顺利拼接成完整样本外预测序列

### D11. 验收标准
- 旧 trainer strategy 不受影响
- 新 strategy 能按轮次正常运行
- test 评估结果与 future prediction 结果清晰分离
- final model 只服务于 future prediction
- 所有轮次的 future prediction 可以按日期无重叠拼接

---

## 7. 功能块 E：接入基于 `predictions.pkl` 的动态选因子

### E1. 目标
在新训练策略中，接入基于 `predictions.pkl` 的历史 RankIC 动态筛因子逻辑。

### E2. 输入文件语义
`predictions.pkl` 中存储了日期以及当日预测的因子 IC / RankIC 信息。  
它不是训练特征本身，而是历史因子有效性的记录。

### E3. 核心筛选规则
在某个重训节点 `t`：
- 只能使用截至 `t` 可见的历史信息
- 从 `predictions.pkl` 中筛出历史 RankIC 大于等于阈值的因子
- 形成这一轮固定使用的因子集合

默认阈值示例：
- RankIC绝对值 >= 0.03

### E4. 需要实现的功能点
1. 能读取 `predictions.pkl`
2. 能根据日期截断到当前节点
3. 能在“只用历史信息”的前提下计算每个因子的历史有效性
4. 能根据阈值筛选出因子
5. 能将该轮筛出的因子传给训练流程
6. 能将该轮实际使用的因子集合保存为 artifact，便于复盘

### E5. 历史信息边界要求
必须严格避免未来信息泄露：
- 不能在第 `t` 天重训时使用 `t+1` 或之后的 RankIC 信息
- 不能先看完整个 `predictions.pkl` 再回头筛

### E6. 轮次更新语义
例如：
- 第一轮在起始训练节点筛因子，得到 factor set A
- 这一轮从训练到 test evaluation 再到 future prediction，都使用 factor set A
- 到下一轮重训节点时，重新筛因子，得到 factor set B
- 下一轮全部阶段使用 factor set B

### E7. 与 feature mask 的关系
建议逻辑为：
- 先确定一个基础候选特征池（例如来自 feature_mask）
- 再用 `predictions.pkl` 的历史 RankIC 在候选池中进行动态筛选
- 最终得到该轮实际训练使用的 feature 子集

### E8. 验收标准
- 动态选因子逻辑可通过 config 启用或关闭
- 开启后，因子集合会随轮次更新
- 同一轮内部不发生因子集合变化
- 仅使用历史信息，无未来泄露
- 每轮实际使用的因子集合可以保存并追踪

---

## 8. 功能块 F：main / evaluator / config / artifact 统一适配

### F1. main.py
需要修改主流程，支持：
- dataloader 之后可选进入 feature_engineering
- preprocess 支持 robust zscore
- trainer 支持 `rankic_refit_roll`
- trainer 在该模式下可接入动态选因子逻辑
- 支持未来预测区间长度配置

### F2. evaluator
需要适配新训练策略的双重语义。

必须能区分：
- 正式 test evaluation
- future prediction 拼接后的统一回测

建议 evaluator 输出至少两类结果：
1. test 阶段指标和图
2. future preds 拼接后的回测结果和图

### F3. artifact 命名
需要保证新功能下的产物命名清晰、可复盘。

至少要避免：
- eval-stage model 和 final-stage model 混名
- test preds 和 future preds 混名
- 不同轮次 selected_factors 混名

建议每轮保存：
- `selected_factors_window_xxx.json`
- `eval_model_window_xxx.*`
- `eval_preds_window_xxx.*`
- `eval_metrics_window_xxx.json`
- `final_model_window_xxx.*`
- `future_preds_window_xxx.*`

最后再保存：
- `future_preds_all.parquet` 或等价文件
- 基于 `future_preds_all` 的统一回测结果

### F4. config 读取与校验
需要让 config 能表达：
- `data.label_postprocess`
- `feature_engineering`
- `preprocess.zscore.method = robust`
- `trainer.strategy = rankic_refit_roll`
- `trainer.rankic_refit_roll.prediction_horizon_days`
- `trainer.rankic_refit_roll.factor_selection`
- `trainer.rankic_refit_roll.rankic_threshold`
- `trainer.rankic_refit_roll.predictions_pkl_path`

---

## 9. 功能块 G：接入已有因子评测系统到 evaluate 模块

### G1. 目标
读取并参考已有的 `run_pred_factor_eval.py`，将其对应的因子评测能力接入当前框架的 `evaluate` 模块中。

接入目标是：
- 对新训练策略 `rankic_refit_roll` 产生的 future prediction 拼接结果做统一因子评测
- 对其他训练策略产生的 prediction 拼接结果也能进入同一套因子评测系统
- 因子评测相关参数可通过 config 控制
- 支持输出 `excess_return=True` 和 `excess_return=False` 两种结果，或只输出其中一种

### G2. 可读取但禁止修改的文件
以下文件只允许读取和调用，**不得做任何修改**：

- `run_pred_factor_eval.py`
- `manage_db_read.pyc`
- `db_settings.json`
- `eval_all.pyc`

这是硬约束，必须严格遵守。

### G3. 接入原则
- 优先复用已有因子评测系统，而不是重写一套新的评测逻辑
- 若直接并入现有 `evaluate` 模块会污染当前框架结构，则允许新增一个单独的评测桥接文件，例如：
  - `ret_pred/evaluate/factor_eval_bridge.py`
  - 或 `ret_pred/evaluate/pred_factor_evaluator.py`
- 但无论采用哪种实现方式，最终都必须能从主流程中被 config 控制调用

### G4. 不同训练策略的评测输入口径
必须明确区分不同训练策略进入因子评测系统的输入来源：

#### 对新训练策略 `rankic_refit_roll`
- 不能使用 test evaluation preds 进入最终因子评测
- 必须使用各轮 `future_preds` 按日期拼接后的完整样本外预测序列
- 拼接结果应保证时间上首尾衔接、不重叠、无未来泄露

#### 对其他训练策略
- 使用该训练策略常规输出的 prediction 结果，按原有逻辑拼接后进入因子评测系统

### G5. 推荐实现方式
建议增加一个中间桥接层，负责完成以下工作：

1. 读取对应训练策略输出的 prediction 文件
2. 根据训练策略类型选择正确的评测输入源
3. 将 prediction 整理成 `run_pred_factor_eval.py` 所需输入格式
4. 调用已有因子评测系统
5. 保存评测结果与图表
6. 记录本次评测的参数配置

推荐新增的桥接函数示例：
- `run_factor_eval_from_preds(...)`
- `build_factor_eval_input(...)`
- `dispatch_factor_eval_by_strategy(...)`

### G6. excess_return 输出要求
因子评测系统需要支持三种模式：

1. 只输出 `excess_return=True`
2. 只输出 `excess_return=False`
3. 同时输出两种结果

如果同时输出两种结果：
- 必须分别保存结果
- 文件名和目录名中必须清晰区分 `excess_true` 与 `excess_false`
- 不得覆盖彼此产物

### G7. config 接入要求
需要在 config 中支持控制以下内容：

- 是否启用已有因子评测系统
- 因子评测输入来源
- 对不同训练策略的输入选择逻辑
- `excess_return` 输出模式：
  - `true`
  - `false`
  - `both`
- 因子评测所需其他参数
- 输出目录与命名方式

### G8. 对 main.py / evaluate 模块的要求
- `main.py` 在训练结束后，应能根据 config 决定是否调用该因子评测系统
- `evaluate` 模块应保留原有评估逻辑，同时新增这一路“因子评测系统接入”能力
- 若采用桥接文件，`evaluate` 负责调度而非重写底层评测实现

### G9. artifact 要求
应清晰区分以下产物：

- 原始模型评估结果
- test evaluation 结果
- future prediction 拼接结果
- 输入已有因子评测系统的整理后因子文件
- `excess_return=True` 的评测结果
- `excess_return=False` 的评测结果

建议命名中显式包含：
- 训练策略名
- prediction source
- excess mode

例如：
- `factor_eval_rankic_refit_roll_future_preds_excess_true/`
- `factor_eval_rankic_refit_roll_future_preds_excess_false/`

### G10. 验收标准
- 不修改任何硬约束文件
- 能成功读取并参考 `run_pred_factor_eval.py`
- 新训练策略可将拼接后的 `future_preds` 接入因子评测系统
- 其他训练策略可将常规 prediction 拼接结果接入同一系统
- `excess_return=True/False/both` 三种模式均可由 config 控制
- 双输出模式下两套结果互不覆盖
- 整体流程中不存在未来信息泄露

---

## 10. 回归验证要求

每做完一个功能块，都应进行最小验证。

### 10.1 dataloader label postprocess
验证：
- 同一日期截面内 `y` 的分布发生变化
- 原始行数不应无故变化
- 分位数截断生效

### 10.2 feature_engineering
验证：
- 新列数正确
- 同一个 stock 的 rolling 特征只使用历史
- 不同 stock 之间不串扰

### 10.3 robust zscore
验证：
- `standard` 与 `robust` 模式可切换
- `robust` 模式使用 median 和 MAD
- 极端值影响应弱于 standard zscore

### 10.4 动态选因子
验证：
- 在给定节点只能看到历史日期
- 满足阈值的因子会被选中
- 每轮 factor set 可以保存
- 同一轮内部 factor set 不变

### 10.5 rankic_refit_roll
验证：
- test 指标来自 eval model
- final model 在 train+test 上重训
- future prediction 可以连续预测 `N` 天
- 不同轮次 future prediction 可以无重叠拼接
- 最终统一回测基于 future prediction 拼接结果，而不是 test prediction

---

## 11. Agent 的输出要求

在完成修改时，agent 应输出：

1. 改了哪些文件
2. 每个文件改动的目的
3. 新增 config 字段说明
4. 新训练策略的数据流说明
5. 动态选因子如何避免未来泄露
6. future prediction 如何拼接成统一回测输入
7. 是否有 breaking change
8. 最小运行示例

如果某一部分暂未完成，必须明确说明，不要假装已经全部支持。