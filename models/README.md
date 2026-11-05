# models/walkforward_v4

生产在用的 LightGBM 排序头（`drop_rank_*.txt`），季度 walk-forward 逐折训练，
共 20 折（测试期 2022Q1 → 2026Q4）。文件名后缀是该折**测试期的起始月份**，
`scripts/daily_signal.py` 取字典序最大的一个（`drop_rank_202610.txt`）作为当日模型。

| 项 | 值 |
|---|---|
| 标签 | `y_rank` = 当日 `open(D)/close(D-1)-1` 的全市场百分比排名 |
| 特征 | 39 个点位内特征（`scripts/train_v4.py` 的 `BASE`），含相对排名类 `cs_*`，不含大盘时序特征 |
| 样本 | 剔除 ST / 退市 / 不可买（涨停、低流动性、`exec_ok=0`）后的全市场 |
| 特征时点 | lag-1，只用 D-1 及更早的 bar，避免未来函数 |
| 规模 | 单折约 670 KB，20 折合计约 13 MB；纯文本，无 pickle 兼容问题 |

**为什么删样本而不加权**：排序头学的是"同一天里谁更强"。涨停、停牌、ST 这些
买不到的票留在训练集里，模型会把它们的次日跳空当成可捕获的 alpha，实盘却一股
也买不到。`drop_` 前缀的一版就是把这个偏差剔掉的结果，也是唯一在用的版本。

复现训练（`scripts/train_v4.py` 同时产出 `keep_rank` / `drop_rank` / `drop_triage` 三组对照）：

```bash
python scripts/train_v4.py --dataset outputs/t1_dataset.parquet \
    --out-dir outputs/walkforward_v4
cp outputs/walkforward_v4/models/drop_rank_*.txt models/walkforward_v4/
```

推理时用 `--models models/walkforward_v4`（默认走 `deploy/config.env` 的 `T1_MODELS`）。
