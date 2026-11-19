# P0 方法与协议差异

本轮均按固定上游 commit 做方法级复现。作者原样本清单未与本机图像逐一恢复，因此不声明精确复现作者分数。补丁清单和 SHA-256 见 `patches/registry.json`；独立源码位于 SSD，原仓库不修改。

## Thomas

保留 HF CLIP 投影特征、logistic regression、原 resize/crop128 和 JPEG 行为。B/16、B/32 与新增 L/14 分别登记，不能混称。训练输入将由新 R/U 清单提供；模型参数传本机固定 revision 目录。特征缓存、全量抽取和训练预算在 P1 实现/验收。R 的 BigGAN 与 BigGAN+ADM 共享相同真实池、相同总原图数和 BigGAN 验证集，GLIDE 仅作最终测试；不是作者样本清单的逐图复刻。

## mixinran

使用 torch 2.6.0 / transformers 4.46.3 的兼容环境，区别于作者 requirements 中的 torch 2.11.0 / transformers 5.19.0；记录为依赖适配，尚不宣称全训练数值等价。

已核对官方 Effort commit `96f5dea2b534d400cfd7003f053c7e93c8e16461` 的 `DeepfakeBench/training/detectors/effort_detector.py`：矩阵位置、rank 1 残差、冻结规则、分类头与当前未启用正交/奇异值约束的行为一致。官方主权重为冻结 Parameter，本地为 buffer，保存形式不同；这不证明完整训练数值等价。参考源码及哈希见 `effort_reference.json`。

当前实现逐层修改 self_attn 的 q/k/v/out 四个 Linear；L/14 的 1024×1024 矩阵固定前 1023 个奇异方向，训练剩余 U/S/V（rank 1），bias 冻结，pooler 后接 Linear(1024,2)。损失为交叉熵，无显式正交/奇异值约束。LoRA 为同位置 rank 8、alpha 8。合成小矩阵测试验证初始输出等价与有限梯度；完整 L/14 SVD、参数统计及 AMP 属于 P1。

VAE 构造器改为固定本地 SD2.1 revision，并用 seed＋父图路径确定每张图片的 posterior RNG。FP16 失败重算使用相同种子。移动数据根会改变路径种子，因此本轮固定路径，并要求派生清单登记父图 ID、seed 和文件哈希；尚未生成 VAE twins。D0/D1/D2 的构造、D3j/D3c 的配对审计仍是对应实验前置项。RAISE/Mirage/EvalGEN 缺失时不运行、也不把简化集命名为完整 D3c。

## xhgSh

删除 val→train 回退与缺失来源静默跳过；统一消费已冻结的清单。模型选择仅使用官方 train 划出的 val，训练入口默认不运行最终测试。修复最后不足 accum 个 batch 时的梯度归一化（原 DataLoader drop_last=True，每个 batch 等大）。合成层测试验证 Adapter/SSF 可训练参数及回载一致性。

U 轨道 seen 为 SD14+BigGAN；R 为 SD14 单源。作者八生成器评估所需 Wukong 缺失，R 完整八源平均仍不可报告。U 预先声明的七源面板不冒充作者八源结果。

逐图评分、多阈值、单分支对照、等价续跑及完整模型 AMP 验收仍在 P1/P2/P3 对应阶段，P0 不将其标记为已运行。

## IAPL

保留原生 transform 和结构；去掉硬编码 CLIP 权重路径，训练及模型选择接新冻结清单，禁止从 test 选 best checkpoint。P0/P1 配置默认不暴露最终测试，原生 TTA 入口暂时显式拒绝运行，直至 P1 完成单 GPU 样本覆盖与逐图参数/优化器完全重置验收；这不是声称已完成 TTA 适配。官方一 epoch 配方作为 R 参照，U 预算待 P1 冻结。完整 TTA/OIS 与无 TTA 消融分开登记。

## 暂不纳入核心 P0 的项目

GAPL 为可选外部参照，未准备 Community Forensics、两阶段训练或完整外部权重。DeiT/Swin/CNNSpot 等附加骨干权重按 P1 所选任务再获取并固定，未因此修改旧环境。所有缺项列在实验矩阵，未从原方法预定分母中删除。
