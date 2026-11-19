# P3 五个未见生成器补充评估

本次分析沿用 2026-11-19 冻结的 P3 模型和各模型的三个阈值。新增 SD1.5、VQDM、Midjourney 各 2,000 张官方 val 生成图，复用 P3 的 ADM、GLIDE 各 2,000 张生成图及同一批 5,000 张真实图。每个模型、每个条件有 15,000 张独立原图，每项单来源二分类指标使用 2,000 张生成图和 5,000 张真实图。每份合并预测均重新计算指标，并逐项检查 ADM、GLIDE 的指标与原始 P3 收据完全一致。

原始 P3 主终点为 ADM、GLIDE 干净原图的等权平均 AUROC。五源结果是 P3 完成后补充的分析。SD1.5 与训练来源 SD1.4 同属 Stable Diffusion 系列。

## 文件

- `by_source.csv` 提供每个模型、seed、条件、阈值和来源的原始指标、实际阈值与混淆矩阵计数。共享真实图在五个来源中各参与一次配对，不能把五行 TN/FP 相加作为独立真实图数。
- `summary.csv` 中的 `macro5` 为等权五源均值，其他行保留单来源。九种退化等权，排除 clean。在每个 seed 内先汇总，然后报告跨 seed 均值和样本标准差。三个核心方法各用 64k 训练图与三个 seed，四个 16k 基线各一个 seed，标准差留空。比例和分数均按 0–1 表示。
- `clean_auroc_contrasts.csv` 提供两项核心配对差值。按原图在真实池与五个生成器内分别有放回抽样，跨模型与 seed 共享抽样权重，对三个固定 seed 取均值后报告 1,000 次配对 bootstrap 的 95% 百分位区间。训练随机性的差异另列三个配对差值与样本标准差。
- 冻结阈值包括固定 0.5（p05）、验证集最大平衡准确率（max_BA）、验证真实图误报率不超过 1%（FPR01）。FPR01 在测试上的实际误报率由数据决定。
- 逐图 CSV、SHA-256、评估协议和完成记录位于 `/mnt/ssd/data/logs/arin5201/reproduction/proposal_p3_five_unseen_20261119/`。清单和哈希来自 `reproduction/manifests/v1_20261114/` 和 `reproduction/proposal/P3_FROZEN.json`。

## 本机复算入口

```bash
bash reproduction/run.sh reproduction/proposal/extend_p3.py
bash reproduction/run.sh reproduction/proposal/five_source_intervals.py
```
