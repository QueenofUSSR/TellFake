# Proposal 首轮（2026-11-15）

配置：`round1.json`；入口：`run.py`。7 个方法按 Frozen CLIP、Effort、Effort+退化、ResNet50、DeiT-S、Swin-T、CNNSpot 顺序执行。seed 42，S1 16,000 原图，完整 validation 10,000 原图，神经网络初始 5 epochs，有效 batch 16。Frozen 特征 batch 32，logistic regression C=1、balanced、max_iter=3000。仅训练/验证，未接入最终测试或 dev。

核心方法采用共同预处理；补充基线沿用 P1 native 输入与增强（含原生 pad/crop），不能将它们解释为仅架构变化。核心无放回 shuffle；补充基线在真假均衡清单上均匀有放回采样（期望类别均衡，不强制每 batch 平衡）。Adam、学习率与 weight decay 来自已验收绑定；StepLR 每 10 epochs ×0.1（首轮不触发）。Effort 无论文约束项，Frozen 为 logistic regression，均需披露。

首轮结束检查完整曲线，不以 seen 指标推断 unseen 泛化。预登记延长上限 15 epochs，在 5/10/15 轮检查；Effort 两组配对。当前入口只执行初始 5 轮，扩展需独立登记，不能改变已完成运行配置。之后再安排单 seed 的 64k 核心规模检查与重复 seeds。

## 启动与恢复

```bash
bash reproduction/run.sh reproduction/proposal/run.py
```

每 250 updates 原子保存 last.pt，每轮保存验证曲线和逐图预测；包含模型、optimizer、scheduler、scaler、RNG、epoch、样本顺序与位置、累计损失。重启同一命令从 last.pt 恢复，完整结果校验指纹后跳过。队列锁避免重复启动；错误写入 status.json 并停止，不静默跳过失败方法。best checkpoint 按完整 clean validation AUROC 选择，严格相等时保留较早轮次。恢复允许重做尚未落盘的步骤。

输出：`/mnt/ssd/data/logs/arin5201/reproduction/proposal_round1_20261115/`
权重：`/mnt/ssd/data/ckpt/arin5201/reproduction/proposal_round1_20261115/`
特征：`/mnt/ssd/data/cache/arin5201/reproduction/proposal_round1_20261115/`

本轮使用完整 checkpoint，未使用 P1 delta 格式；磁盘空间已检查（启动前约 348 GiB 可用）。P0/P1 原有结果不变。小样本 smoke 产物放在独立 `_smoke` 目录，仅用于功能验收。

`curve.json` 保存逐轮加权损失、整体/分来源指标、学习率、更新次数；自动 convergence 字段仅作粗筛，最终判断需要完整曲线。`result.json` 记录最佳 checkpoint、阈值、耗时和产物哈希。指标独立从保存的预测 CSV 重算、验证数量及阈值预测后才标记完成。

## 配对延长至 10 epochs（2026-11-17）

用户已授权，配置 `extension10.json`。从首轮两组各自的第 5 轮 last checkpoint 续训，完整恢复训练状态。原首轮产物不变，延长输出存储在各根目录下的 `proposal_extension10_20261117`。`continuation_provenance.json` 保存来源哈希和完整状态读回验证。

```bash
bash reproduction/run.sh reproduction/proposal/run.py --config reproduction/proposal/extension10.json
```

仅执行 Effort 两组，总计 10 epochs；best 在第 1–10 轮选取。追加训练耗时记录不包含前 5 轮；最终测试不变。准备脚本 `prepare_extension10.py` 只用于一次性登记，不重复执行；恢复使用上述运行命令。

## Seen validation 九种退化（2026-11-17）

登记配置 `validation_degradations.json`，入口 `evaluate_degradations.py`。Frozen CLIP 使用首轮分类器；Effort 使用 10-epoch 实验的第 10 轮 best，增强版使用第 8 轮 best。所有方法评估相同 10,000 validation 原图，包含 Real 5,000、SD14/BigGAN 各 2,500。九种单操作退化在共同模型预处理之前执行，JPEG subsampling=2，缩放为 PIL bicubic；三种阈值均从各模型 clean validation 原样复用。

共 270,000 条预测，每方法/条件写独立 CSV 和 JSON，逐图重新计算整体与分来源指标并校验数量。以条件为单位断点恢复；有完整校验收据的条件跳过。运行先复核 checkpoint 哈希并重放 64 张 clean 预测。未访问最终测试，不进行任何参数更新、阈值拟合或 checkpoint 重新选择。

```bash
bash reproduction/run.sh reproduction/proposal/evaluate_degradations.py
```

输出：`/mnt/ssd/data/logs/arin5201/reproduction/proposal_validation_degradations_20261117/`。

## S2 64k 核心规模检查（2026-11-17）

`scale64k.json` 登记三组核心 seed 42，Effort 均从原始预训练初始化训练 10 epochs（40,000 updates），Frozen 新抽取 S2 特征并拟合。`run_scale.py` 相对冻结的 `run.py` 仅将训练数量断言改为配置值，并切换默认配置；原入口和历史产物不变。

```bash
bash reproduction/run.sh reproduction/proposal/run_scale.py
```

输出/权重/缓存位于各自根目录下的 `proposal_scale64k_20261117`；失败恢复使用相同命令。S1 与 S2 在同 epoch 预算下比较，更新次数不同；不声称纯数据量收益。最终测试封存。

## 64k clean 对比与退化 validation（2026-11-17）

`compare_scales.py` 校验两种规模全部 6 个核心产物，并从同一批 10,000 张 clean validation 预测独立重算指标，输出 `clean_scale_comparison.json`。

64k 九种退化配置：`validation_degradations64k.json`；入口：`evaluate_degradations64k.py`，与 16k 入口仅配置文件名不同，使用相同退化、预处理与指标实现。Frozen 使用 64k 分类器，Effort best 为第 6 轮，增强版为第 9 轮。沿用各自 clean validation 的三个阈值，不重新拟合。输出 `proposal_validation_degradations64k_20261117`；按条件支持恢复。最终测试未使用。

```bash
bash reproduction/run.sh reproduction/proposal/evaluate_degradations64k.py
```
