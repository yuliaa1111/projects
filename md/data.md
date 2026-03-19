# ret_pred 数据契约与数据语义说明（新版）

## 1. 文档目的

本文档用于描述 `ret_pred` 框架实际依赖的数据假设与新增后的数据语义。

目标是让 AI coding agent 明确理解：
- 规范内部数据格式是什么
- label 是怎么构造和后处理的
- feature 是怎么筛选和扩展的
- preprocess 是怎么工作的
- 新训练策略下数据流、选因子、预测和回测语义如何区分
- 重构和扩展时哪些数据语义必须保留

这份文档描述的是**运行时数据契约**，不只是文件名说明。

---

## 2. 规范内部数据格式

框架规范的内部格式是一个 **long-format DataFrame**。

每一行表示一个股票在某个日期上的一个样本：
- 一行 = 一个 `(date, stockid)` 对应的样本

典型列包括：
- `date`：交易日
- `stockid`：股票标识
- `y`：可选标签列
- feature columns：数值型因子 / 风险 / 风格 / 市场特征
- optional passthrough columns：原始数据中可能存在、但不作为模型特征的透传列

重要约束：
- `(date, stockid)` 理想情况下应当唯一
- 所有训练、预处理、feature engineering、cut、评估逻辑都默认基于这种面板式 long-format 数据
- 即使原始输入来自 wide parquet，内部处理也必须统一到 long-format

---

## 3. 支持的原始输入形式

### 3.1 Long parquet
单个 parquet 文件，已经是 long format。

典型列包括：
- `date`
- `stockid`
- 可选 label 列
- 大量 feature 列
- 可选价格列，例如 `open_1d`

### 3.2 Wide parquet 目录
一个目录中，每个字段单独保存为一个 wide parquet：
- index = date
- columns = stockid
- values = 该字段的取值

这些 wide 矩阵会在内部被转换为 long-format。

---

## 4. 核心列定义

### 4.1 关键列
框架中会反复使用以下关键列：
- `date_col`，默认：`date`
- `stockid_col`，默认：`stockid`
- `label_col`，通常：`y`

### 4.2 标签列
标签列通常命名为 `y`。

`y` 可能来自：
1. 原始数据中已经存在的 label 列
2. 基于价格列在加载阶段动态构造
3. 进一步变换后的 label，例如 normalized label、ranked label、extreme label
4. label postprocess 后得到的最终训练标签

### 4.3 用于构造 label 的价格列
某个价格型原始列可用于构造未来收益标签，例如：
- `open_1d`
- 未来也可能支持 `close_1d`

当前常见用法是：
- 基于 `open_1d` 构造未来一期收益
- 概念上通常对应 next-day open-to-open return
- y label约束：每日的raw y为ret(t+1),即[open(t+2)/open(t+1)]-1,不能用次日open/当日open，这样得到的其实是当日ret，必须用次次日open/此日open，这样得到的才是次日ret。具体构造方式如下，不允许擅自啊改变。

    """
    在 long-format 数据上构造“下一期收益率”label：
      y_t = P_{t+2}/P_{t+1} - 1

    - 对每只股票按日期排序，然后 groupby(stockid) 得到：
      p_t1 = shift(-1), p_t2 = shift(-2)
    - 支持简单收益 / 对数收益

    注意：
    - 本函数不会删除最后一天的样本；最后一天会得到 NaN label（由后续 preprocess 决定如何处理）
    - label_method（open_to_open 等）在本模块中仅作为 meta 记录

    Args:
        df_long: 输入 long DataFrame，至少包含 [date_col, stockid_col, price_col]
        date_col: 日期列名
        stockid_col: 股票 ID 列名
        price_col: 价格列名（例如 open_1d）
        label_name: 生成的 label 列名
        log_return: True 生成 log return；False 生成 simple return

    Returns:
        pd.DataFrame: 返回包含新增 label 列的 DataFrame（copy）

    Raises:
        ValueError: price_col 不存在
    """
    if price_col not in df_long.columns:
        raise ValueError(f"price_col '{price_col}' not found in long df")

    out = df_long.copy()

    # 确保 shift 的“下一期”是按时间递增对齐
    out = out.sort_values([stockid_col, date_col])

    g = out.groupby(stockid_col, sort=False)[price_col]
    p_t1 = g.shift(-1)
    p_t2 = g.shift(-2)
    if log_return:
        out[label_name] = np.log(p_t2) - np.log(p_t1)
    else:
        out[label_name] = p_t2 / p_t1 - 1.0

    return out
---

## 5. Feature 筛选逻辑

框架支持通过 `feature_mask_csv` 进行特征筛选，同时需要支持`feature_mask_csv`为null时也能运行。

### 5.1 为什么存在这个机制
其目的包括：
- 保证训练和预测使用一致的特征集合
- 可以通过外部文件管理因子筛选结果
- 更新 feature 版本时不需要改代码

### 5.2 支持的 CSV 形式
当前支持至少两种格式。

#### 格式 A
两列表形式，例如：

| feature | use |
|---------|-----|
| f1 | 1 |
| f2 | 0 |
| f3 | 1 |

#### 格式 B
单行宽表形式：列名是特征名，值是 0/1。

### 5.3 重要行为
feature mask 的逻辑在训练和预测阶段都要共用。
这是保证特征对齐的关键。
需要支持`feature_mask_csv`为null时也能运行
---

## 6. Label 语义
框架需要支持切换不同的y label，一共支持这几种 y：
y_raw：原始未来收益标签。
y continuous without rank：连续收益标签。
y continuous with rank：按日横截面 rank 后的连续标签。
y normalized continuous：归一化后的连续标签。
y extreme without rank：直接按连续值做 top/bottom extreme 标签。
y extreme with rank：先 rank 再做 top/bottom extreme 标签。

### 6.1 连续型 label
默认形式：
- 未来收益作为连续数值型目标

典型行为：
- 基于某个配置的价格列计算下一期收益
- 可以是 simple return，也可以是 log return，取决于配置

### 6.2 归一化 label
当前框架支持 label normalization。

原始收益标签可以按近期波动率或近期平均绝对收益做缩放。  
这部分逻辑会直接改变 target 语义，不能在重构中悄悄改掉。

### 6.3 Extreme label
框架支持 extreme-label 风格。  
可以按横截面把样本映射成 top / bottom 两端标签，中间部分要么丢弃，要么赋中性值。

### 6.4 Ranked label
框架支持按日期横截面排序得到 ranked label。  
target 可以从原始 future return 变成截面排序后的 rank-style label。

### 6.5 Label postprocess（新增）
在 raw label 构造完成之后，框架必须支持可选的 label 后处理。

第一版要求支持的处理链为：
1. 按交易日对 `y` 做横截面 zscore
2. 按交易日对 `y` 做横截面 winsorize

winsorize 采用分位数截断，默认参数为：
- lower_q = 0.05
- upper_q = 0.95

重要约束：
- 这是对 label 的后处理，不是对价格列做处理
- 默认顺序是先 zscore，再 winsorize
- 必须显式通过 config 启用，不能默认静默生效
- 最终参与训练的 `y` 应当是 postprocess 之后的结果
- 如需保留原始 `y`，应通过 config 控制是否另存一列，例如 `y_raw`

---

## 7. Feature Engineering 阶段（新增）

在 dataloader 与 preprocess 之间，可以插入专门的 feature engineering 阶段。

目标是基于原始因子列生成更多时间统计特征，以补充单时点截面特征的信息不足。

### 7.1 第一版要求支持的统计特征
- rolling mean
- rolling std

### 7.2 默认窗口长度
- 5
- 10
- 20

### 7.3 计算规则
rolling 统计特征必须满足以下约束：
- 按 `stockid` 分组
- 按 `date` 升序排序
- 只能使用当前时点及过去历史
- 不能泄露未来信息
- 生成的新特征追加到原始特征集合中

### 7.4 与 feature mask 的关系
第一版建议逻辑为：
- 先确定原始 feature 列集合
- 再对这些原始 feature 生成 rolling 统计特征
- 新生成特征自动加入模型输入特征集合

---

## 8. Preprocess 语义

预处理必须明确区分：
- 训练阶段的 `fit_transform`
- 预测 / 推理阶段的 `transform`

### 8.1 基础修正
典型预处理一开始会做：
- 把 `date` 解析为 datetime
- 将 `inf` 和 `-inf` 替换为 NaN

### 8.2 feature 列推断
如果没有显式给出 feature 列，框架会自动从数值列中推断特征列，同时排除：
- 日期列
- 股票代码列
- 标签列
- 可选透传列

### 8.3 缺失值策略
当前至少有两类关键缺失值语义：
- `strict`
- `tree_friendly`

重构时必须保留这样一个事实：缺失值处理与模型家族相关。

### 8.4 Winsorization
框架支持对特征做 winsorize。

### 8.5 标准 z-score
框架支持传统 z-score 标准化。

### 8.6 Robust z-score（新增）
除了传统 z-score 外，preprocess 还必须支持稳健 zscore。

稳健 zscore 使用：
- `median`
- `MAD`（median absolute deviation）

典型形式为：

`z = (x - median) / (MAD + eps)`

如有需要，也可以支持是否乘 `0.6745` 的可选参数，但必须通过 config 控制。

重要约束：
- robust zscore 应被视为 zscore 的一种 method，而不是完全独立、重复的一套流程
- 默认不替代传统 zscore，只有在 config 中指定 `method: robust` 时才启用
- 可按 `date` 截面计算

### 8.7 预处理状态保存
训练阶段的 preprocess 可以保存 state 文件。  
预测阶段应复用这个 state。

---

## 9. 日期切分与 Rolling Windows

框架采用 rolling training，而不是单次静态切分。

### 9.1 Split 的概念
一个时间区间会被切成多个 rolling windows。  
在每个 rolling window 内，再继续划分为：
- train
- valid
- test

### 9.2 重要语义
切分本质上是**按日期切**，而不是按行随机切。  
这是硬约束，因为任务本质是时间序列 / 面板预测。

---

## 10. 模型输入语义

本项目的传统流程中，会将 preprocess 后的 long-format 数据进一步转换成模型可直接使用的输入。  
但在新的训练策略下，更重要的是明确“每轮使用哪一套因子”和“预测的是 test 还是 future block”。

### 10.1 Tree family
树模型通常需要：
- 二维表格特征
- 根据模型与 preprocess 策略决定 NaN 是否保留
- 使用该轮固定下来的特征集合

### 10.2 Sequence family
神经网络 / 序列模型通常需要：
- 稠密数值数组
- 更严格的 NaN 处理
- 同样使用该轮固定下来的特征集合

### 10.3 轮次内特征集合必须固定
新训练策略下，最重要的约束之一是：
- 因子集合允许在不同重训轮次之间更新
- 但同一轮模型从训练到对应 future prediction 期间，特征集合必须固定
- 不允许在同一轮测试 / 预测区间内按天更换特征集合

---

## 11. predictions.pkl 的语义（新增）

`predictions.pkl` 文件中保存了历史日期以及对应因子的 IC / RankIC 信息。

该文件在新训练策略中的作用不是直接作为模型输入，而是作为**滚动选因子的历史依据**。
predictions.pkl 的本质含义可以理解成：
每个交易日 -> 当天一组因子的 prediction 分数
也就是：
外层 key：日期
内层 Series.index：因子名
内层 Series.values：该日期这些因子的预测值

### 11.1 使用原则
- 在每个新的重训节点，只能使用截至该节点及以前的历史信息
- 不能读取未来日期的因子表现
- 可按给定阈值筛选有效因子，例如 RankIC绝对值 >= 0.03
  目标语义
  1. 每个窗口取一个锚点日期（“窗口第一天”）。
  2. 只看 predictions.pkl 在这个日期当天的 rankic。
  3. 当天满足 abs(rankic) >= 0.03 的因子入选该窗口。
  4. dataloader 预读因子池 = 所有窗口入选因子的并集。

trainer 接法应该是这样的：
对于每个 rolling window：
先拿到这个窗口第一天 window_start_date
调 select_factors_by_day_rankic(pred_map, select_date=window_start_date, threshold=0.03, candidate_pool=当前可用字段)
得到该窗口固定因子集 selected_factors
用这批因子跑该窗口 train和test，然后再把模型在train+test的全量数据上fit一遍得到final mode
用final model预测future，在future的第一天，读取当日predictions.pkl中符合要求的因子集，用新因子集预测
所以每个window要在“train的第一天”和“future的第一天”重选因子，涉及的因子都要进入select_factors_by_day_rankic
下一窗口再重复此步骤

### 11.2 输出含义
在某个重训节点得到的筛选结果，是**这一轮固定使用的因子集合**。

这一轮的以下阶段都必须使用这同一套因子：
- train
- test evaluation
- refit final model
- future block prediction

直到进入下一轮重训，才允许重新筛一次因子并更新集合。

---

## 12. 新训练策略下的数据语义（新增）

新增训练策略建议命名为 `rankic_refit_roll`。

该策略下需要明确区分四类数据阶段。

### 12.1 历史选因子阶段
在某个重训节点 `t`：
- 读取截至 `t` 可见的历史因子表现
- 基于 `predictions.pkl` 筛出满足阈值的因子
- 得到这一轮固定使用的 factor set

### 12.2 eval 阶段
在同一轮中：
- 用该 factor set 在 train split 上训练模型
- 用该模型在 test split 上做正式样本外评估

该阶段的 test 指标：
- 是这一轮正式的 out-of-sample evaluation
- 应作为 evaluator 的正式 test 结果
- 不得被后续 refit 结果覆盖

### 12.3 refit 阶段
eval 阶段完成后：
- 将 train + test 合并
- 用同一套 factor set 在合并数据上重新训练 final model

该 final model：
- 不用于回填 test 指标
- 只用于预测 test 之后的 future block

### 12.4 future prediction 阶段
final model 在该轮中用于：
- 连续预测未来 `N` 天
- `N` 由 config 控制
- 这些 future prediction 才是最终拼接回测的输入

### 12.5 轮次滚动
若本轮预测了未来 `N` 天，则：
- 下一轮的重训节点向前滚动 `N` 天
- 重新基于截至新节点的历史信息筛选因子
- 用截至新节点的全部历史样本重训
- 再预测新的 future block

### 12.6 最终回测语义
最终回测不应基于每轮内部的 test 预测拼接，而应基于：
- 各轮 future block prediction 的时间拼接结果

test 评估的作用是：
- 评估这一轮模型的样本外泛化能力

future prediction 的作用是：
- 构建最终连续样本外预测序列并做统一回测

---

## 13. 评估阶段的数据语义

本项目的 evaluation 同时具有两类性质：
- 监督学习预测评估
- 因子风格的市场评估 / 回测

在新训练策略下，evaluator 必须明确区分：
1. 真实 test evaluation
2. future block prediction 拼接后的统一回测

这两者不能混为一谈。

---

## 14. Artifact 语义

框架会把输出写入 run 目录。

新增功能后，artifact 语义应进一步明确：
- label postprocess 若启用，可按需保存处理后标签或中间列
- feature engineering 若启用，可按需保存增强后的特征表或中间 cache
- `rankic_refit_roll` 策略下，应区分：
  - 当前轮使用的 factor set
  - eval-stage model
  - eval-stage preds
  - eval-stage metrics
  - final-stage model
  - future block preds
  - 全部 future preds 拼接后的统一回测输入
  
## 14A. 已有因子评测系统接入语义（新增）

框架需要支持将 prediction 输出进一步送入已有的因子评测系统。

complete_factor_analysis(
    factor_name='factor_name',  # 自定义因子名称
    input_factor_df=df,  # 如果已存在因子DataFrame，可直接传入
    start_date=datetime(2023, 1, 1, 0, 0, 0),
    end_date=datetime(2024, 12, 31, 0, 0, 0),
    frequency=1,  # 数据频率，默认日频 '1d'
    factor_direction='positive', # 'positive' or 'negative'
    portfolio_type='long_only',
    long_groups=[9, 10],
    short_groups=[1, 2],
    group_num=10,
    neutralize=False,  # 是否进行中性化处理
    save_figures=False, # 是否保存图表
    save_dir=None,  # 保存结果的目录
    return_data=False,
    verbose=True,
    client=client,  # 使用之前创建的数据库连接对象
    excess_return=False, # 是否使用超额收益率进行分析
    eval_price='open', # close to close or open to open
    enable_turnover_fee=True, # 是否启用费率模式
    turnover_fee_rate=0.00045
)

### 对新训练策略 `rankic_refit_roll`
进入因子评测系统的输入必须是：
- 各轮 `future_preds` 按日期拼接后的完整样本外预测序列

不能使用：
- test evaluation preds

因为 test evaluation 的语义是模型评估，
而 future prediction 的语义才是最终连续样本外信号生成。

### 对其他训练策略
进入因子评测系统的输入为：
- 该策略常规 prediction 输出按日期拼接后的结果

### excess_return 语义
因子评测系统必须支持：
- `excess_return=True`
- `excess_return=False`
- 同时输出两种结果

双输出时，两套结果必须分开保存并明确区分。
---

## 15. 对 Agent 的硬约束

未经明确指示，不得修改以下核心设定：

1. long-format 作为规范内部表示
2. 按日期切分的时间序列语义
3. 基于 feature-mask 的特征对齐机制
4. 训练阶段 fit 与预测阶段 transform 的分离
5. 支持 raw return、normalized return、ranked label、extreme label、label postprocess 等多种 label 语义
6. 支持新增的 feature engineering 阶段
7. 支持传统 zscore 与 robust zscore 两种 preprocess 语义
8. 新训练策略中动态选因子只能使用历史信息
9. 同一轮训练到预测期间特征集合必须固定
10. 最终统一回测必须基于 future prediction 的拼接结果，而不是 test prediction

---

## 16. 总结

核心思想如下：

原始数据输入形式可以不同，但一旦进入框架内部，统一标准应当是：
- long-format 面板数据
- 明确的 label 语义
- 可选的 label postprocess
- 可选的 feature engineering
- 明确可复用的 preprocess state
- 按日期进行 rolling split
- 在重训节点按历史 RankIC 动态筛选因子
- 同一轮固定 factor set 完成训练、评估、重训与 future prediction
- 最终拼接 future prediction 形成完整样本外预测序列并统一回测