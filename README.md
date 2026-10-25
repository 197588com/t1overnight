# t1overnight — A股 T+1 隔夜选股模型

在 A 股十年硬数据池（5,577 只 / 2016-01 ～ 2026-10 / 10,888,307 根日 K）上，
按"每日 1 票、次日卖出"的 T+1 结构重新建模、回测，并给出诚实的成本边界。

> 结论摘要见 `outputs/report/T1隔夜模型报告.html`。
> **一句话：横截面 alpha 是真的（日均超额 +0.174%，t=2.08），但每日全换仓 0.20% 的往返摩擦
> （年化 ≈49%）把它吃光，净期望为负。**

## 目录结构

```
t1overnight/
├── t1overnight/
│   ├── features.py      # 41 个点位内特征 + 标签（numpy 逐票计算，10.9M 行 8 秒）
│   └── backtest.py      # 可买过滤、选股、成本模型、绩效指标
├── scripts/
│   ├── build_dataset.py     # 数据池 → 建模数据集（含 exec_ok 停牌标记）
│   ├── train_v2.py          # 季度 walk-forward，双头（排序头 + 概率头），--lag 1 出严格口径
│   ├── run_backtest_v2.py   # 变体对比、分位结构、IC、成本敏感性、盈亏平衡
│   ├── hold_period.py       # 持有期 1/2/3/5 日对照
│   ├── predict_next.py      # 用最新模型给下一交易日打分（严格口径，只需 D−1 数据）
│   └── report.py            # 生成单文件 HTML 报告
└── outputs/                 # 数据集、预测、回测明细、live 名单、报告
```

## 复现步骤

```bash
PY=/Users/yanglin/.workbuddy/binaries/python/envs/default/bin/python
SRC=/Users/yanglin/Desktop/未命名文件夹/datapool/parquet    # 十年数据池 parquet 目录

$PY scripts/build_dataset.py --source "$SRC" --out outputs/t1_dataset.parquet
$PY scripts/train_v2.py --dataset outputs/t1_dataset.parquet --out-dir outputs/walkforward_v2
$PY scripts/run_backtest_v2.py --pred outputs/walkforward_v2/predictions.parquet
$PY scripts/hold_period.py
$PY scripts/report.py
```

严格口径对照（特征只到 D−1）：

```bash
$PY scripts/train_v2.py --lag 1 --out-dir outputs/walkforward_v2_lag1
$PY scripts/run_backtest_v2.py --pred outputs/walkforward_v2_lag1/predictions.parquet \
    --out-dir outputs/backtest_v2_lag1
```

每日增量（只做推理，秒级）：

```bash
$PY scripts/predict_next.py     # -> outputs/live/scores_snapshot.csv
                               #    outputs/live/blacklist_next_session.csv
```

## 关键设计（踩过的坑）

1. **标签必须可执行**：D+1 停牌的样本要剔（`exec_ok`），否则"下一根 bar"可能是几个月后的复牌跳空。
2. **剔除大盘特征**：第一版把 `mkt_ret / mkt_up_ratio / mkt_amt_ratio` 喂进排序模型，
   gain 前三全被它们占据，排序能力退化——单票选择只需横截面信息。
3. **只在可买样本上训练**：涨停、低流动性、ST 代理要从训练集剔掉，否则模型学会一堆买不到的票。
4. **排序头 > 概率头**：预测"涨多少的相对次序"比预测"涨不涨"更贴收益，毛收益 0.144% vs 0.129%。
5. **ST 只能用代理**：数据源的 name 是当前名，回溯会引入未来函数，改用涨跌停制度反推（5% 制度识别）。

## 成本口径（10 万单票）

| 口径 | 佣金 | 印花税 | 滑点 | 双边合计 |
|---|---|---|---|---|
| 保守 | 万 2.5 ×2 | 0.05% | 0.05% ×2 | **0.20%** |
| 优化 | 万一 ×2 | 0.05% | 0.02% ×2 | **0.10%** |
| 极优 | 万一 ×2 | 0.05% | 0 | **0.07%** |

盈亏平衡点 ≈ 0.144%（Top1 日均毛收益）。0.07% 口径年化约 +5%，0.20% 口径年化约 −23%。

> 研究记录，不构成投资建议。
