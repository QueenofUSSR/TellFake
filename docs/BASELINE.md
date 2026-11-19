> 历史记录：本文件描述旧 ResNet/v3 实验，过程中的“待运行”等状态仅代表当时情况。当前工作以 `plan.md` 和 `reproduction/README.md` 为准。

# AI-Generated Image Detection

基于本机 GenImage 子集的 ResNet 二分类实验，覆盖数据清理、BigGAN/SD1.5 训练、
ADM/GLIDE 未知生成器测试，以及 JPEG、缩放、模糊鲁棒性评估。

旧 ResNet 基线运行参考；当前任务以 [实施计划](../IMPLEMENTATION_PLAN.md) 为准。

## 运行环境与存储

使用 Docker；不在宿主安装 CUDA/PyTorch。镜像仅安装依赖，代码、数据和结果通过 bind mount 挂入。

| 内容 | 宿主路径 | 容器路径 |
| --- | --- | --- |
| 代码 | `/mnt/ssd/code/arin5201` | `/workspace` |
| 数据，**只读挂载** | `/mnt/ssd/data/datasets/GenImage` | `/data/genimage` |
| 新划分清单 | `/mnt/ssd/code/arin5201/data/manifests/v2` | `/workspace/data/manifests/v2` |
| 模型 checkpoint | `/mnt/ssd/data/ckpt/arin5201` | `/ckpt` |
| 实验记录与评估结果 | `/mnt/ssd/data/logs/arin5201` | `/logs` |
| Torch 权重缓存 | `/mnt/ssd/cache/torch` | `/cache/torch` |

环境基于现有 `assignment2:torch2.5.1-cu124-v1`：PyTorch 2.5.1+cu124、
torchvision 0.20.1+cu124；额外依赖锁定于 `requirements-runtime.txt`。
每个训练/评估目录记录实际镜像 ID、运行参数、软件版本和源代码快照。

在项目目录执行：

```bash
./scripts/build_runtime.sh
./scripts/run.sh test
```

新镜像为 `arin5201:torch2.5.1-cu124-v1`；构建时仅发送 Dockerfile 和依赖清单。
如需其他路径，可设置 `GENIMAGE_DATASET_DIR`、`GENIMAGE_CHECKPOINT_DIR`、
`GENIMAGE_LOG_DIR`；默认路径全部位于工作 SSD。

## 1. 检查、去重并划分

```bash
./scripts/run.sh prepare
```

默认读取 `standard_v1`，完整解码每张图片，并同时计算文件 SHA-256 与 RGB 像素 SHA-256。
相同像素内容仅保留一份；标签冲突、跨来源相同图片、损坏图片和样本不足会直接报错。
原始图片、旧 `manifests/v1` 不作修改；已存在的输出目录不会被覆盖。

默认划分协议：

- 已见生成器：**BigGAN、SD15**。各自来源 `train` 的 80% 用于训练、20% 用于验证；
  来源 `val` 全部保留给 `seen_test`。
- 未见生成器：**ADM、GLIDE**，各随机抽取 1,000 张用于 `unseen_test`，不参与训练或调参。
- 真实图片先全局去重，再从同一个池中**无放回**分配到四个集合；每个集合真假数量相等。
- 四个集合的路径、文件哈希、像素哈希均必须互不重叠。
- 默认 seed=42。CSV 中 `split` 是本次实验划分，`source_split` 才是来源目录中的 train/val。

对当前 20,000 张本机图片，预期得到：

| 划分 | 真实图片 | AI 图片 | 总数 |
| --- | ---: | ---: | ---: |
| train | 3,200 | BigGAN 1,600 + SD15 1,600 | 6,400 |
| val | 800 | BigGAN 400 + SD15 400 | 1,600 |
| seen_test | 1,000 | BigGAN 500 + SD15 500 | 2,000 |
| unseen_test | 2,000 | ADM 1,000 + GLIDE 1,000 | 4,000 |

另有 1 张完全重复图片被排除、5,999 张唯一图片未选用。以生成的
`split_config.json` 为准。`duplicates.csv` 记录去重关系，`unused.csv` 记录剩余图片。

这是**第三方小规模子集上的自定义协议**，保留的是本机来源目录的 train/val 关系，
不能将结果直接称为官方完整 GenImage benchmark。
当前去重覆盖字节/像素完全重复，不保证消除不同裁剪或有损重编码的近似重复。

## 2. 训练与单图预测

完整基线训练：

```bash
./scripts/run.sh train --config configs/resnet50_baseline.json --name resnet50_baseline_s42
```

默认：ImageNet 预训练 ResNet-50，224×224 输入，batch size 64，10 epochs，
AdamW，学习率 1e-4，cosine scheduler，混合精度。
首次使用会将约 98MB 的 ResNet-50 权重下载到工作 SSD 上的 Torch 缓存。

标签为 `0=Real`、`1=AI`。使用 BCEWithLogitsLoss；基础增强仅包含随机水平翻转。
真假图片使用同一个预处理过程，不把文件名、目录或 generator 标签输入模型。

只使用验证集选取模型（AUROC，平分时比较 BCE loss）和分类阈值（balanced accuracy）。
测试集只作来源隔离检查，不参与模型或阈值选择。

输出位置：

```text
/mnt/ssd/data/ckpt/arin5201/resnet50_baseline_s42/best.pt
/mnt/ssd/data/ckpt/arin5201/resnet50_baseline_s42/last.pt
/mnt/ssd/data/logs/arin5201/resnet50_baseline_s42/
  config.json, runtime.json, source/, source_sha256.json
  train_used.csv, val_used.csv, manifest_sha256.json
  history.jsonl, best_validation.json, best_val_predictions.csv
```

checkpoint 是推理权重及实验元数据，不包含用于断点续训的优化器状态。
`--name` 必须唯一，重复名称会报错。

单图预测：

```bash
./scripts/run.sh predict \
  --checkpoint /ckpt/resnet50_baseline_s42/best.pt \
  --image /data/genimage/standard_v1/ADM/train/ai/0000141.jpg
```

返回标签、`p_ai`、预测类别的置信分数及实际阈值；概率尚未校准，不等同于真实性保证。
checkpoint 内包含预处理尺寸，推理时自动复用，不会重新下载预训练权重。

## 3. 跨生成器与鲁棒性评估

```bash
./scripts/run.sh evaluate \
  --checkpoint /ckpt/resnet50_baseline_s42/best.pt \
  --name resnet50_baseline_s42_eval
```

同时评估 `seen_test` 和 `unseen_test`。默认 8 种条件：

| 条件 | 操作 |
| --- | --- |
| clean | 原图 |
| jpeg:90 / jpeg:70 / jpeg:50 | 指定质量重新编码 JPEG |
| resize:0.5 / resize:0.25 | 按原始图片尺寸降采样，再恢复原尺寸 |
| blur:1 / blur:2 | Gaussian blur，Pillow radius=1 / 2 |

扰动应用于 RGB 原图，然后执行与验证阶段相同的 resize、归一化。
每种条件对真假图片一视同仁。全部测试固定使用 checkpoint 保存的验证集阈值。

结果位于 `/mnt/ssd/data/logs/arin5201/resnet50_baseline_s42_eval/`：

- `metrics.json`：完整指标、checkpoint/manifest 指纹和评估协议。
- `summary.csv`：整体及各生成器对比，可直接用于制表。
- 每个“划分 × 条件”的预测 CSV：图片路径、标签、生成器、P(AI)、预测类别。

指标包括 Accuracy、Balanced Accuracy、Precision、Recall、F1、AUROC、AP、真实图片误判率
及混淆矩阵计数。每个生成器都与**同一组真实测试图片**配对计算指标，另报告生成器平均
Balanced Accuracy；保留样本数量，避免把仅含 AI 的子集当成完整二分类评估。

可以只运行特定条件：

```bash
./scripts/run.sh evaluate \
  --checkpoint /ckpt/resnet50_baseline_s42/best.pt \
  --name resnet50_clean_eval --conditions clean
```

提供了对称增强对照配置，可在同一划分上比较：

```bash
./scripts/run.sh train --config configs/resnet50_robust.json --name resnet50_robust_s42
./scripts/run.sh evaluate --checkpoint /ckpt/resnet50_robust_s42/best.pt --name resnet50_robust_s42_eval
```

`robust` 在训练时以相同概率对真假图片应用 JPEG（p=0.5，Q=50–100）、
缩放（p=0.5，比例 0.25–0.99）、模糊（p=0.3，radius=0.1–2.0）。
也可用 `--augmentation jpeg` / `resize` / `blur` 单独消融。
这些增强是否改善泛化需以正式实验结果判断。

## 小规模联调

以下流程只用于检查代码和 GPU 链路，模型及结果都会标记 `smoke_test=true`，
不能当作项目检测性能：

```bash
./scripts/run.sh train --config configs/resnet50_baseline.json \
  --name smoke_baseline --epochs 1 --batch-size 16 --workers 2 \
  --max-train-samples 64 --max-val-samples 32

./scripts/run.sh evaluate --checkpoint /ckpt/smoke_baseline/best.pt \
  --name smoke_baseline_eval --batch-size 16 --workers 2 --max-samples 64

./scripts/run.sh predict --checkpoint /ckpt/smoke_baseline/best.pt \
  --image /data/genimage/standard_v1/ADM/train/ai/0000141.jpg
```

代码只实现检测器训练，不训练或运行图像生成模型，也不读取 `tiny_genimage` Parquet。
固定输入尺寸与统一 JPEG 编码不代表消除了所有来源偏差；报告中仍应讨论分辨率、
内容分布和预训练数据影响，以及本机来源数量和年代的限制。
