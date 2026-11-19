# P1/S0 执行协议与偏离

S0 只作方法接口与成本测量，不作正式方法优劣结论。使用已冻结 U S0 的 4,000 张原图；验证采用 P0 validation 中预先固定的 1,024 张（512 real、SD14/BigGAN 各 256）。没有将最终 test 清单加载进模型数据管线，也未运行最终测试。seed=42，训练方法一 epoch；冻结特征法为一次完整抽取与 logistic regression 拟合。

## 运行规则

六组冻结特征配置使用 FP32、batch 32：B/32 resize、B/16 resize/crop128 × JPEG 开关、L/14 resize。四组 Thomas B/16 的 crop/JPEG 分布保持原定义；随机增强以 seed/epoch/original_id 定位，区别于作者顺序 RNG 流，便于跨 worker 和断点回放。原生 extract 在固定验证图上独立核对特征一致性。特征缓存键包含模型固定路径、清单指纹、预处理、seed 与实现哈希。

训练方法 batch 16。原生 Effort/LoRA 保留作者 common_aug、Adam、学习率和 weight decay，使用类均衡有放回采样；因此 4,000 次呈现不代表每张训练图都被抽中，单独记录实际独立原图数。统一 Effort 基础版只用原生 pad/random crop 与水平翻转；退化版只增加十选一（clean 或九个 proposal 操作），JPEG420、bicubic 缩放，发生于 crop 前，两类完全相同。它们是新增 U 对照，不命名为完整 D3c。

双分支及单分支保留原生 resize/crop/flip 与模型定义；单分支为相应 encoder 加单 linear logit。双分支仍是两个分支 logits 线性融合。S0 主测使用 BF16 autocast，区别于 xhgSh 原脚本默认 FP16；这项精度差异必须披露，R 轨道若需严格保留 FP16 要额外验收。IAPL 按官方 GenImage gate/condition/smooth 开启，使用原生损失、Adam、FP32 和原生图像预处理。

所有神经网络先做 FP32/BF16 有限前后向、可训练参数核查、磁盘 checkpoint＋optimizer/RNG 一步回放、最终 checkpoint 的预测一致性；检查后恢复初始状态才进入实际 S0 epoch。恢复记录保留 model、optimizer、scheduler、scaler、Python/NumPy/CPU/CUDA RNG、完整采样顺序和 next_batch；增强由原图/epoch/seed 决定，不依赖 worker 调度。队列遇失败停下，修复后明确重跑；本轮不声称已完成 P2 的无人值守中断恢复队列。

## 成本口径

前十 batch 为预热，之后冻结特征法测 115 batch，训练方法测 240 batch。计时前后 CUDA synchronize，计时包括 next(loader)、CPU→GPU、前向、反向（若有）、optimizer update 和有限性检查；checkpoint IO 不混入每步吞吐，epoch wall time 另报。验证为相同 1,024 张、固定 batch，无 warm-up 剔除。首次模型加载/SVD 和 checkpoint 大小另报。

Effort 原生 CPU SVD 的结果按原权重字节、rank 和源码哈希缓存；不改变 SVD 公式、不换成随机低秩近似。首次成本与缓存命中初始化成本分别记录。其 frozen weight_main 为 buffer，不能仅用 PyTorch parameters() 数量冒称模型变小；汇总同时报告注册参数数与浮点 state 元素数。

## IAPL TTA/OIS

仅在 S0 验证子集里抽取固定 16 real＋16 fake，每图 clean/JPEG70 两条件；原生 32 views、2 adaptation steps、selection_p=.2、AdamW lr=.005，单 GPU 完整覆盖 64 个图像/条件样本。用本地 no_sync 空上下文替代 DDP 包装，不改变 entropy/OIS 公式。

每个图像/条件开始前重置全部可适配参数、所有 buffers、optimizer 状态和该图 RNG；不跨图累积学习。A→B→A 回放必须预测完全一致；冻结参数在整段运行前后哈希一致。每次两次适配前向/反向，另一次 OIS 推理，记录视图呈现数和完整每图时延（包含加载、增强、重置）。OIS-off 只作接口探针。32 图阈值只是诊断，正式低 FPR 阈值必须在完整验证集按相同 TTA 流程重新冻结。

## 已知边界

RAISE/Mirage/EvalGEN/Wukong 缺项、GAPL 以及附加非 CLIP 骨干不在本轮 16 配置的核心 S0 完成声明中。没有生成 VAE twins，也没有完成 D0→D3c 复现。不存在作者分数的逐图精确复现主张。

首次队列启动因文件名 queue.py 遮蔽 Python 标准库产生递归启动，已停止本任务的启动进程并改名 run_queue.py，增加 main guard 与 import 无副作用测试；尚未进入模型训练的原始日志保留。后续运行只允许一个 GPU 队列，失败日志不删除。

## 补充骨干与 checkpoint 验收

核心 16 配置后追加 Effort-NPR、NPR-only、Effort-RGB、无辅助损失四组及 ResNet50/DeiT-S/Swin-T/CNNSpot 四个 proposal 骨干，共 24 个训练/特征配置，另有 IAPL TTA 探针。附加权重见 extra_weights.lock.json。补充八组 S0 保留原生模型/损失/common_aug，以固定无放回顺序遍历 4,000 张；S1/S2 按原生训练配方恢复类均衡有放回抽样。这项采样差异已预声明，S0 数字不用于比较性能。

为避免为 PEFT 每次重复保存冻结骨干，新增 verified_delta_v1 格式。保存可训练权重、全部可变 buffer（包括不进入 state_dict 的 nonpersistent buffer）、optimizer/scheduler/scaler/RNG/sampler；遗漏的冻结权重逐张量哈希验证。逐方法从原生预训练基座重建，检查所有 state 张量与完整 checkpoint 完全一致，并在故意破坏保存组件后回载检查预测完全一致。首次测试抓出 position_ids 非持久 buffer 未恢复，第二次抓出标量 buffer 哈希兼容性；已修复并增加单元测试，失败日志保留。本轮完整 checkpoint 仍保留，正式队列预算采用验证后的 delta 格式。

独立消融检查发现首版 JPEG 与 crop 共用 RNG、U 退化开关会推进 crop/flip RNG；已分别拆为 seed+1000 的 JPEG 流、seed+2000 的退化选择流，crop/flip 流不变。相关三组重新训练/抽取，旧结果移入 attempts/*_rng_v1，旧源码按 SHA-256 保存在 source_versions；预算只用修正后结果。加入两项独立 RNG 测试。

独立逐图重算发现 NumPy float32 标量比较会将 FPR 阈值的 float64 nextafter 边界舍入，导致六个 Frozen 配置各一个 CSV prediction 值与已正确计算的汇总指标不一致。统一用 Python float64 比较；保留原 CSV、阈值和分数，重新生成 prediction 列并增加边界测试，详见 prediction_precision_fix.json。TTA 最终探针按 16 Real＋8 SD14＋8 BigGAN 覆盖两 seen 来源，初版探针保留在 attempts。
