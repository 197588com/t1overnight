# t1overnight — A 股 T+1 隔夜选股策略

每日 1 票、开盘买、次日开盘卖。信号在 **D 日收盘后**用截面模型打分产生，
**D+1 开盘买入、D+2 开盘卖出**，全程只持有一个隔夜。

> 研究项目，**不是**已验证的盈利策略。回测见下文「诚实的数字」，其中包含了
> 为什么它在成本面前很脆弱。任何实盘使用前请自己重跑一遍。

---

## 1. 策略结构

| 时点 | 动作 | 用到的数据 |
|---|---|---|
| D 日收盘后 | 计算 39 个截面特征，给全市场打分，取第 1 名 | D 日及更早的日 K（特征 lag-1） |
| D+1 09:15–09:25 | 看集合竞价开盘价：高于 `昨收 × 1.015` 则**当日放弃**（不顺延次选） | 开盘价 |
| D+1 开盘 | 按开盘价买入，仓位 = 本金 × 40%，整百股向下取整 | — |
| D+2 开盘 | 无条件卖出 | — |
| 全程 | 不设止损、不设止盈、不看盘中 | — |

选股逻辑是**截面排序**而非择时：模型只回答"同一天里谁明天开盘更强"，
不预测大盘方向。这也是特征里刻意剔除大盘时序变量的原因。

---

## 2. 数据

本地 parquet 数据池（`T1_DATAPOOL`），需要以下文件：

| 文件 | 内容 |
|---|---|
| `daily_k.parquet` | 全市场日 K：OHLC、成交量额、流通股本、换手 |
| `universe.parquet` | 可交易标的清单与板块归属 |
| `fin_quarterly.parquet` | 季度财务（ROE、营收/净利同比），用于 3 个特征 |

规模参考：5,577 只 × 2016-01 ～ 2026-10，约 1,089 万根日 K。

**三源互证**（`scripts/fetch_today.py`）：tushare + 腾讯 + 新浪，不设固定主源。
两源收盘价一致即定案，凑不齐或对不上才拉第三家；参考源与其余各源都不一致时弃用参考源。
完整性判据（任一命中即判不完整）：行数 < 5000 / 比前一交易日少 5% 以上 /
流通股本缺失超 5%。**新鲜度硬校验**：数据集最后一天必须等于目标交易日，
否则不出信号——数据晚一天仍会"成功"输出昨天的票，这是本策略最危险的失效模式。

---

## 3. 特征与模型

39 个特征（`scripts/train_v4.py` 的 `BASE`），全部为点位内/截面量：

| 组 | 数量 | 例 |
|---|---|---|
| 动量与形态 | 9 | `ret_1/3/5/10/20/60`、`gap`、`intraday`、`close_pos`、`amp` |
| 波动与位置 | 10 | `vol_5/20`、`vol_ratio`、`ma_dev_5/20/60`、`dist_high/low_20`、`dist_high_5` |
| 量能 | 5 | `log_amt`、`amt_ratio`、`amt_trend`、`turn`、`turn_ratio` |
| 规模与炒作 | 4 | `log_mktcap`、`lu_cnt_5`、`lu_cnt_20`、`streak_up` |
| 财务 | 3 | `roe`、`net_profit_yoy`、`revenue_yoy` |
| 截面相对 | 5 | `cs_ret_1/5`、`cs_amt`、`cs_turn`、`cs_ma_dev_20` |
| 其它 | 3 | `z_ret`、`days_listed`、`board` |

模型：LightGBM `LGBMRegressor`，标签是当日 `open(D)/close(D-1)-1` 的全市场
百分比排名（`y_rank`），即排序学习。季度 walk-forward，共 20 折，
每折只用该折测试期之前的数据训练，产出 out-of-sample 预测。

**只在可买的样本上训练**：涨停、停牌、ST、低流动性、`exec_ok=0` 全部剔除。
留在训练集里会让模型把买不到的票的次日跳空当成 alpha，回测漂亮、实盘一股买不到。

生产模型已随仓库提供：`models/walkforward_v4/drop_rank_*.txt`（20 折，纯文本）。

---

## 4. 诚实的数字

Walk-forward out-of-sample，每日 Top1，2022-01 ～ 2026-10，n = 1161 个交易日：

| 指标 | 值 |
|---|---|
| 截面 IC（日频） | 0.0768 |
| ICIR | 8.17 |
| Top1 日均毛收益 | **+7.9 bp** |
| Top1 t 值 | **0.67** |
| 生产口径往返成本 | 13 bp（万 1 佣金 ×2 + 印花税 5bp + 滑点 2bp ×2） |

结论要分开看：

- **排序能力是真的**：IC 0.077 / ICIR 8.2 在横截面上稳定，不是噪声。
- **Top1 的收益不显著**：t = 0.67，毛收益 7.9bp 低于 13bp 成本，净期望为负。
  "排序准"和"第一名够赚"是两件事，中间差着 Top1 与 Top10 的衰减。
- **成本是主要敌人**：日频全换仓，13bp 往返 ≈ 年化 32% 的摩擦。
  历史上另一版口径（20bp 往返）下年化拖累约 49%，直接把 alpha 吃光。

样本量约束（单笔净收益标准差 σ ≈ 267bp）：要分辨 25bp 的 edge 需要约 894 笔；
60 笔的 95% 置信区间约 ±67bp，**永远验不出 25bp**。任何"胜率提升了"的说法，
在样本不足时都不成立。

---

## 5. 目录结构

```
t1overnight/
├── t1overnight/              # 库：特征、可买过滤、回测内核
│   ├── features.py           #   特征与标签（numpy 逐票，千万行级可用）
│   ├── features_v3.py        #   日内/市场类扩展特征
│   ├── backtest.py           #   tradable() 过滤、选股、成本模型、绩效指标
│   ├── live.py               #   生产环境取数与快照
│   └── replay.py             #   历史重放
├── scripts/
│   ├── daily_run.py          # 生产总入口：取数 → 重建 → 出信号 → 结算
│   ├── daily_signal.py       # 打分与 TopN 选股
│   ├── fetch_today.py        #   tushare/腾讯/新浪 三源互证取当日 bar
│   ├── build_dataset.py      # 数据池 → 建模数据集
│   ├── train_v4.py           # 季度 walk-forward，三组对照（keep/drop/rank/triage）
│   ├── run_backtest_v2.py    # 回测、分位结构、成本敏感性、盈亏平衡
│   ├── settle_ledger.py      # 账本结算（D+2 开盘价、13bp 成本）
│   ├── ledger_attr.py        # 收益三段归因 A/B/C
│   ├── board_tiers.py        # 板块与分层分析
│   ├── weekly_review.py      # 周度复盘
│   ├── push_wecom.py         # 可选：企业微信推送（默认关闭）
│   └── ...
├── deploy/
│   ├── config.env.example    # 配置模板（复制为 config.env 后改）
│   ├── run_daily.sh          # launchd 入口，含失败告警与日志轮转
│   ├── status.sh             # 面板：今日信号、账本、待卖清单
│   ├── render_launchd.sh     # 从 config.env 渲染 plist
│   ├── install_launchd.sh    # 装载/卸载定时器
│   └── config_check.py       # 配置自检
├── models/walkforward_v4/    # 生产模型（20 折 drop_rank）
├── TRADING_SPEC.md           # 作业规范：口径、风控、统计规则
└── outputs/                  # 数据与产物（不入库，见 .gitignore）
```

---

## 6. 快速开始

```bash
git clone https://github.com/197588com/t1overnight.git
cd t1overnight
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

cp deploy/config.env.example deploy/config.env
$EDITOR deploy/config.env          # 至少改 T1_PY 与 T1_DATAPOOL
```

重建数据集并训练（耗时以小时计，需要完整数据池）：

```bash
python scripts/build_dataset.py --source "$T1_DATAPOOL" --out outputs/t1_dataset.parquet
python scripts/train_v4.py --dataset outputs/t1_dataset.parquet --out-dir outputs/walkforward_v4
```

用已有模型直接出信号（跳过重建，秒级）：

```bash
python scripts/daily_signal.py \
    --dataset outputs/t1_dataset.parquet \
    --models models/walkforward_v4 \
    --out-dir outputs/live
```

完整生产跑（取当日数据 → 重建 → 出信号 → 结算 → 可选推送）：

```bash
python scripts/daily_run.py            # 当晚，截止 21:00
python scripts/daily_run.py --morning  # 次日晨补，截止 08:40
```

定时（可选，macOS）：

```bash
sh deploy/render_launchd.sh     # 从 config.env 渲染 17:24 / 07:54 两个备用定时器
sh deploy/install_launchd.sh    # 装载
sh deploy/status.sh             # 面板
```

---

## 7. 账本与口径

唯一收益真源：`outputs/live/forward_ledger.csv` 的 `net_ret`（已扣 13bp 往返）。

| 行状态 | 含义 | 计入胜率 | 计入均值 |
|---|---|---|---|
| `filled` | 已买已卖 | 是 | 是 |
| `pending` | 已买未卖 | 否 | 否 |
| `gap_skipped` | 开盘高于上限，未成交 | 否 | 否 |
| `void` | 板块权限外，无效信号 | 整行剔除 | 整行剔除 |

S+2 行情未到期时禁止用 S+1 收盘顶替；跳空放弃的假想收益只进参数评估、不写回账本。
完整规则（含收益三段归因、样本量鉴别力、调参纪律）见 `TRADING_SPEC.md` §4.4。

---

## 8. 已知限制

1. **Top1 收益不显著**（t = 0.67），且毛收益低于往返成本；策略当前不具备统计意义上的正期望。
2. **无止损止盈**，唯一退出是"D+2 开盘卖"。极端行情下没有下行保护。
3. **滑点假设乐观**：13bp 往返要求万 1 佣金 + 2bp 单边滑点；小市值票开盘冲击成本可能远高于此。
4. **模型为 2026-10 定格**，季度重训需要完整数据池；不重训会随时间衰减。
5. **数据依赖私有数据池**：`daily_k.parquet` 需自行准备，仓库不含行情数据（体量与版权原因）。
6. **ST 识别用代理**（涨跌停制度反推），因为数据源的 name 是当前名，直接回溯会引入未来函数。

---

## 9. 免责

仅供研究与技术交流，不构成任何投资建议或收益承诺。据此操作，风险自负。
A 股 T+1 交易、涨跌停、停牌、印花税与佣金规则都可能变化，回测结论随之失效。

许可：代码 MIT；`models/` 下的模型文件随同 MIT。
