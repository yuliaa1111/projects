# ret_pred 重构蓝图（新版）

## 1. 项目概述

本仓库是一个用于股票收益预测与 alpha 信号生成的研究框架。

该框架服务于以下典型研究场景：
- 基于 long-format（长表）股票因子数据进行建模
- 采用 rolling / walk-forward 的时间滚动训练方式
- 使用 YAML 配置文件驱动数据、预处理、切分、训练、预测、评估全流程
- 同时支持树模型与神经网络模型
- 同时支持回归任务与分类 / 伪分类任务
- 训练后保存预测结果，并进一步做因子风格评估与全市场评估

这不是一个从零开始的新项目。  
这是一个在已有可运行框架基础上的**重构 + 功能扩展**任务。

本次工作的目标包括两部分。

第一部分是**结构重构**：
1. 提升代码可维护性
2. 提升模块边界清晰度
3. 提升 config schema 的一致性
4. 除非明确要求，否则尽量保持当前行为与语义不变
5. 让项目更适合“本地用 Codex 改代码 → 上传 GitHub → 服务器 pull 下来运行”的工作流

第二部分是**新增功能**：
1. 在 dataloader 中增加对 `y` 的按日 zscore + winsorize 处理，并通过 config 可插拔
2. 新增 feature engineering 模块，对 `x` 增加 rolling mean / rolling std 等时间统计特征，并通过 config 可插拔
3. 在 preprocess 中增加稳健 zscore（基于 median + MAD），并通过 config 可插拔
4. 新增一种训练策略：先 train→test 评估，再用 train+test 重新 fit final model，最后预测 test 之后的未来一段区间
5. 新增基于 `predictions.pkl` 的历史 RankIC 因子筛选逻辑，用于滚动重训时动态更新因子集合
6. 支持自定义每轮单次预测区间长度 `N`，并将各轮 next-window 预测拼接后统一回测

---

## 2. 当前目标工作流

目标工作流如下：

1. 在本地机器上维护代码和文档
2. 让 Codex / AI agent 先阅读 `blueprint.md`、`data.md`、`tasks.md`、`config_design.md`
3. agent 根据文档在本地修改代码
4. 我在本地检查代码与文档修改结果
5. 将仓库 push 到 GitHub
6. 在远程研究服务器上 pull 最新代码
7. 在服务器上运行训练 / 预测 / 评估

因此，本次重构和新增功能都应优先服务于以下目标：
- 本地开发迭代清晰顺畅
- 远程服务器执行可复现
- 尽量减少环境相关的隐式假设
- 便于通过日志和落盘产物排查问题
- 便于后续继续扩展 label、loss、feature engineering、训练策略、动态选因子和评估逻辑

---

## 3. 重构原则

本项目应当**基于现有框架重构**，而不是彻底从头重写。

### 核心原则
除非有充分理由，否则应尽量保留当前 pipeline 的语义、数据流和实验口径。

### 需要保留的内容
- long-format 作为核心内部数据结构
- YAML 驱动的配置方式
- rolling / walk-forward 的训练工作流
- tree 与 sequence 两类模型输入支持
- 当前的 label 构造逻辑与预处理概念
- 基于 run 目录的实验产物落盘方式
- 训练后继续做 predict / evaluate 的工作流
- 当前评估体系的核心语义

### 可以重点优化的内容
- 目录结构
- 命名一致性
- config schema 组织方式
- builder / registry 模式
- 模块之间的接口设计
- 重复逻辑
- 过长、职责混杂的文件
- 训练态与推理态边界不清的问题
- 当前文档不够适合 AI agent 理解的问题

### 不应该发生的事
- 不要悄悄修改数据语义
- 不要悄悄修改 label 定义
- 不要悄悄修改评估指标与评估口径
- 不要把项目替换成一个通用 ML 模板
- 不要为了“代码更干净”就删掉当前支持的重要工作流
- 不要因为新增功能而破坏现有训练 / 评估结果的可复现性

---

## 4. 当前主流程与目标主流程

当前逻辑主链路大致为：

`main.py`
→ 读取 config
→ 初始化日志
→ 加载 long-format 数据
→ 按需构造 label
→ 数据预处理
→ 生成 rolling windows / split dates
→ 将 long-format 数据切成模型可用输入
→ 在滚动窗口上训练模型
→ 保存模型 / 预测 / history
→ 运行评估
→ 按需执行训练后的推理

在新增功能后，目标主链路建议更新为：

`main.py`
→ 读取 config
→ 初始化日志
→ dataloader 加载数据并按需构造 label
→ dataloader 按需执行 label postprocess（y 的 zscore / winsorize）
→ feature_engineering 生成 rolling 统计特征
→ preprocess 对特征做清洗 / 标准化 / 缺失值处理
→ split / windows 生成滚动窗口
→ trainer 按指定策略完成 train→test eval→refit→future predict
→ evaluator 评估 test 阶段与统一回测结果
→ predictor / post-train inference 按需输出结果

这里的 trainer 不再只是“单次 train 后 test”，而是需要支持：
- 在 test 上做正式样本外评估
- 用 train+test 重新拟合 final model
- 使用截至当前时点可见的信息重新筛选因子
- 按自定义预测区间长度预测未来 N 天
- 将所有 future prediction 拼接成完整样本外序列用于统一回测

---

## 5. 当前主要模块

当前项目中的主要模块包括但不限于：

- `main.py`：程序入口与总调度
- `dataloader.py`：加载 long-format 或 wide parquet 数据，构造 label，筛选特征
- `preprocess.py`：数据清洗、winsorize、z-score、缺失值策略、fit/transform 状态管理
- `split.py`：日期切分逻辑
- `windows.py`：滚动窗口生成
- `cut/`：把 long-format DataFrame 转成 tree 或 seq 的模型输入
- `tree_models/`：LightGBM、XGBoost、CatBoost 的封装 / builder
- `nn_models/`：Linear、LSTM、GRU、Transformer 的封装 / builder
- `losses/`：可插拔 loss registry 与 objective 映射
- `trainer/`：滚动训练、调参、模型选择、插件逻辑
- `predictor/`：滚动预测或单独推理
- `evaluate/`：评估、比较、全市场因子风格评估
- `paths.py`：路径渲染与实验产物目录管理
- `utils/logger.py`：日志模块

本次新增后，需要增加或扩展以下职责：

- `feature_engineering.py`：新增，对原始特征生成时间统计特征
- `dataloader.py`：扩展，支持 label postprocess
- `preprocess.py`：扩展，支持 robust zscore
- `trainer/`：扩展，支持新的带动态选因子滚动策略
- `main.py`：扩展，支持 feature_engineering 阶段与新 trainer 策略
- `evaluator/`：适配新训练策略下的 test 评估、next-window 预测拼接与统一回测
- 读取 `predictions.pkl` 的因子筛选逻辑：新增，供 trainer 在重训节点调用

---

## 6. 目标架构原则

重构后的项目在功能上应与当前版本大体一致，但模块边界更清晰。

目标原则：
- 一个模块只做一类事情
- 尽量减少隐藏副作用
- 数据契约明确
- 状态对象明确
- 更容易测试
- 更容易从 config 追踪到运行行为与输出产物

一个理想中的逻辑分层可以是：

- 数据输入层：`dataloader`
- 特征增强层：`feature_engineering`
- 预处理层：`preprocess`
- 时间切分层：`split` / `windows`
- 训练策略层：`trainer`
- 推理层：`predictor`
- 评估层：`evaluate`
- 因子筛选层：`factor_selection` 或 trainer 内部独立子模块
- 工具层：`paths` / `logger` / `registry` / `utils`

不要求一定重构成完全新的目录结构，但要求职责边界更清楚。

---

## 7. 必须保持不变的约束

以下内容是硬约束。

### 7.1 规范内部数据格式
规范内部数据格式是 long-format 面板数据：
- 一行 = 一个 `(date, stockid)` 样本
- 列通常包括：
  - date
  - stockid
  - 可选 label
  - feature 列
  - 可选透传列 / 元信息列

### 7.2 配置驱动行为
框架必须继续保持 YAML 驱动。
关键行为应通过配置控制，而不是写死在代码里。

### 7.3 训练 / 推理解耦
训练阶段的 fit 逻辑与预测阶段的 transform 逻辑必须保持清晰分离。
预处理状态必须能在预测时复用。

### 7.4 支持 rolling 工作流
框架必须继续支持：
- rolling windows
- 每个 window 内部的 train / valid / test 划分
- 基于 run 目录的产物保存

### 7.5 产物持久化
输出结果必须继续方便在磁盘上检查，包括：
- logs
- preprocess 输出
- preprocess state
- 训练模型
- predictions
- evaluation 结果
- sweep summaries

### 7.6 特征与标签语义不能被悄悄改掉
包括但不限于：
- label 构造方式
- label 的后处理方式
- 特征筛选逻辑
- 特征标准化方式
- 评估口径
- rolling 窗口定义
- 动态选因子的历史信息边界

---

## 8. 当前版本的主要问题

以下是当前版本中已知、需要在重构中改善的问题：

1. 一些文件过长，同时混合了 orchestration 和业务逻辑
2. config 字段越来越多，分组方式不总是一致
3. 训练 / 预测 / 不同模型家族之间存在部分重复逻辑
4. 当前框架已支持较多选项，但接口理解成本越来越高
5. 一些模块之间命名不统一
6. 现有文档更适合人类阅读，不够适合 AI coding agent
7. 本地路径与服务器路径的环境假设混在一起
8. 当前如果继续扩 classification、custom loss、feature engineering、training strategy、动态因子筛选，会越来越难维护

---

## 9. 本次重构与扩展范围

### 包含在范围内
- 代码结构重组
- 类型标注和数据契约改进
- config 解析逻辑优化
- builders / registries 结构优化
- 将过长函数拆成更小单元
- 统一 artifact 路径管理
- 统一 train / predict / eval 接口
- 提升后续实验开发的可读性
- 保持本地 + GitHub + 服务器 workflow 顺畅
- 增加 label postprocess
- 增加 feature engineering 模块
- 增加 robust zscore
- 增加新的 trainer strategy
- 增加基于 `predictions.pkl` 的历史 RankIC 动态筛因子
- 支持自定义单轮未来预测长度，并拼接回测

### 默认不在范围内，除非明确要求
- 改变核心量化研究目标
- 重新设计策略思想本身
- 用外部通用框架替换现有工程
- 删除当前 tree 或 sequence 工作流支持

---

## 10. 新增功能要求

### 10.1 在 dataloader 中增加 y 的按日 zscore + winsorize 处理
在 label 构造完成后，框架必须支持对 `y` 进行按交易日横截面处理。

第一版必须支持的处理链为：
1. 按交易日对 `y` 做 zscore 标准化
2. 按交易日对 `y` 做 winsorize 截断

winsorize 使用分位数截断，默认上下分位数为：
- lower_q = 0.05
- upper_q = 0.95

此功能必须：
- 放在 `dataloader` 模块中
- 通过 config 控制是否启用
- 支持可插拔
- 明确区分“原始 label 构造”和“label postprocess”

### 10.2 新增 feature engineering 模块，对 x 增加 rolling 统计特征
需要新增专门的特征工程模块，用于在原始因子基础上构造时间统计特征。

第一版必须支持：
- rolling mean
- rolling std

默认窗口长度包括：
- 5
- 10
- 20

这些 rolling 特征必须：
- 按 `stockid` 分组
- 按 `date` 排序
- 只能使用当前时点及过去历史，不能使用未来信息
- 生成后追加到原始特征集合中
- 最终作为模型输入

### 10.3 在 preprocess 中增加稳健 zscore
preprocess 模块中，除了传统的均值 / 标准差 zscore 外，还必须支持一种更稳健的标准化方式。

第一版要求支持：
- 基于 `median`
- 基于 `MAD`（median absolute deviation）

该功能必须：
- 作为 preprocess 中 zscore 的一种新 method
- 通过 config 控制是否启用
- 默认不替代传统 zscore，只有在 config 指定时才使用
- 支持按 date 截面计算

### 10.4 新增训练策略：先评估，再 refit，再预测未来区间
需要新增一种滚动训练策略，建议命名为：

`rankic_refit_roll`

该策略的核心思想是：
- 在每个重训轮次中，先用固定因子集合完成 train→test 的正式样本外评估
- 然后在 train+test 合并样本上重新训练 final model
- 再用 final model 去预测 test 之后的未来一段区间
- 所有轮次的 future prediction 最终按时间拼接，形成完整的样本外预测序列并统一回测

### 10.5 基于 predictions.pkl 的动态选因子
新训练策略必须支持基于 `predictions.pkl` 中保存的历史因子表现进行动态选因子。

基本规则：
- `predictions.pkl` 中存有日期以及当日预测因子的 IC / RankIC 信息
- 在每个新的重训节点，只能使用截至该节点当日及以前的历史信息
- 按给定阈值筛出 RankIC 大于等于某阈值（例如 0.03）的因子
- 这一轮模型从训练到对应 future prediction 都固定使用这一套因子
- 下一轮重训时才允许更新因子集合

### 10.6 支持自定义未来预测区间长度
不能默认每次只预测 1 天。  
必须支持配置单轮未来预测区间长度 `N`。

执行语义为：
- 模型在某个重训节点完成后，连续预测未来 `N` 天
- 下一轮时间窗口向前滚动 `N` 天
- 各轮预测区间首尾衔接、不重叠
- 最终能够顺利拼接成完整样本外预测序列并用于统一回测

### 10.7 接入已有因子评测系统
框架需要接入已有的 `run_pred_factor_eval.py` 所代表的因子评测系统。

接入要求：
- 新训练策略 `rankic_refit_roll` 必须使用拼接后的 `future_preds` 作为因子评测输入
- 其他训练策略使用其 prediction 拼接结果作为评测输入
- 相关参数必须通过 config 控制
- 必须支持：
  - 仅输出 `excess_return=True`
  - 仅输出 `excess_return=False`
  - 同时输出两种结果
- 以下文件为只读依赖，禁止修改：
  - `run_pred_factor_eval.py`
  - `manage_db_read.pyc`
  - `db_settings.json`
  - `eval_all.pyc`

如直接接入现有 `evaluate` 模块会造成污染，可以增加单独的桥接文件，但必须保持主流程可调度。
---

## 11. 兼容性要求

尽量保持向后兼容，尤其是以下内容：
- 关键 config 概念
- run 目录的语义
- prediction 保存格式
- preprocess state 的复用方式
- 核心 train / predict 命令
- 旧训练策略仍可继续使用

如果必须引入 breaking change：
1. 明确说明原因
2. 写清迁移步骤
3. 尽量保留旧行为
4. 避免一次同时改变多个语义

---

## 12. 建议的实施顺序

### Phase 1：文档与契约
- 补全文档
- 明确 data contract
- 明确 config schema
- 明确新增功能挂载在哪一层

### Phase 2：数据层改造
- dataloader 扩展 label postprocess
- 新增 feature_engineering 模块
- preprocess 扩展 robust zscore

### Phase 3：训练链路改造
- trainer 增加 `rankic_refit_roll`
- 接入基于 `predictions.pkl` 的动态选因子
- 支持 future horizon 预测与拼接
- main.py 接入新策略
- evaluator 适配新产物语义

### Phase 4：验证与样例
- 增加 example config
- 跑 smoke test
- 验证旧流程不被破坏
- 验证新流程能正常产出结果

---

## 13. 对 AI Agent 的行为要求

在修改代码时，agent 应当：
- 先阅读 `blueprint.md`
- 再阅读 `data.md`
- 再阅读 `tasks.md`
- 再阅读 `config_design.md`
- 优先做增量式重构，而不是大规模破坏性重写
- 修改代码时尽量保持已有语义不变
- 对新增功能的实现保持模块边界清晰
- 每一步都保证代码尽量仍可运行
- 对新增 config 字段保持命名一致和层级清晰
- 对 artifact 命名保持可复盘、可追踪
- 明确区分 test evaluation 与 future prediction backtest 的语义

agent 不应该：
- 擅自脑补新的数据假设
- 悄悄删掉当前支持的重要路径
- 为了“简化项目”而删掉量化特有逻辑
- 把当前 workflow 替换成玩具级 demo pipeline
- 因为新增功能而重写整个项目

---

## 14. 成功标准

如果满足以下条件，就算本次重构与扩展成功：

1. 项目更容易在本地理解和修改
2. Codex / agent 在改代码时更不容易误解项目结构
3. 代码可以顺畅 push 到 GitHub，并 pull 到服务器运行
4. 旧训练流程仍能运行
5. 新增功能能通过 config 正常启用
6. 动态选因子只使用历史信息，无未来泄露
7. 新训练策略下 test 评估结果与 future prediction 回测语义清晰区分
8. 所有轮次的 future prediction 能拼接成完整样本外回测序列
9. 后续继续扩展 label、feature engineering、trainer strategy、factor selection 的成本更低

---

## 15. Agent 当前的直接任务

你当前的直接任务不是重新设计量化策略。

你当前的直接任务是：
- 理解当前项目
- 找出稳定接口
- 在现有框架基础上重构模块边界
- 增加 label postprocess
- 增加 feature_engineering
- 增加 robust zscore
- 增加 `rankic_refit_roll` 训练策略
- 接入基于 `predictions.pkl` 的动态选因子
- 支持 future horizon 预测与统一回测
- 保持当前行为尽量不变
- 提升可维护性
- 让仓库更适合“本地 Codex → GitHub → 远程服务器”这一工作流