> 历史记录：本文件描述旧 ResNet/v3 实验，过程中的“待运行”等状态仅代表当时情况。当前工作以 `plan.md` 和 `reproduction/README.md` 为准。

# v3 数据与训练协议（2026-11-12）

2026-11-13更新：缺失来源已按独立补充清单冻结为Midjourney2,000张与COCO4,995张，详情见 [SUPPLEMENT_V3](SUPPLEMENT_V3.md)。下面关于缺口的描述保留为v3初始冻结时的历史范围；v3本身未改动。

入口：`scripts/prepare_v3.py`。数据根目录为 `/mnt/ssd/data/datasets/GenImage/official_v3`，清单为 `data/manifests/v3`。清单已生成，不得覆盖或原位修改。重建需另取版本。seed=42。

训练 16,000 张（Real 8,000，BigGAN/SD15 各 4,000）；验证 7,000 张（Real 5,000，两种 AI 各 1,000）；开发 4,000 张（Real 2,000，两种 AI 各 1,000）。上述全部来自官方 train。

已见测试 BigGAN/SD15 各 1,000 张 AI，未知测试 VQDM 2,000 张 AI，共享独立的 5,000 张真实测试池。全部来自官方 val。真实池不可跨生成器累计为独立样本。未运行最终预测，不用最终测试选择模型或阈值。

当前是缩减的最终测试范围：Midjourney 2,000 张及独立外部真实照片 5,000 张尚缺，不能宣称已完成原计划的双未知来源与外部误报评估。Midjourney 的来源身份预留为盲测，未来补充须使用另行冻结的补充清单，保持现有清单及训练选择不变。GLIDE/ADM 曾用于旧基线，不作为全新未知来源。

## 审计范围与去重

原始库存登记来源为下载目录结构和归档镜像登记；标签不是独立的内容真实性认证。候选来自 BigGAN、SD15 的 train/val 和 VQDM val AI，以及各包共享的 ImageNet nature 池。真实图片按 ImageNet 文件原始身份全局分组，即使重新编码也不跨集合；若同一身份跨官方 train/val，优先保留 val。

先全局按文件哈希去重，并保守排除旧 v2 全部清单中已有的文件与像素内容，再随机抽取 50,000 张候选。候选全部完整解码，记录原始尺寸、格式、文件 SHA-256、经过 EXIF 转向的 RGB 像素 SHA-256 和 dHash。

近似检索为 64 位 dHash、汉明距离不超过 4；五个不重叠哈希段保证该距离内候选被检索。共 337 对候选，连通分组后每组最多保留一张；混合真假或跨 AI 生成器的组件共 182 张全部排除。候选不等于真实重复或标签冲突。按固定 seed 抽查 12 对，包含明显近重复，也有大量不同图片；维持保守排除策略。没有逐一人工认证全部候选。

像素和近似审计只覆盖实验候选，不是对约 134 万张储备的全面近似去重。该方法不能保证检出所有裁剪或派生图。像素、字节、路径、原图身份与分组 ID 在六个清单间均检查无重叠。

审计产物位于 `/mnt/ssd/data/logs/arin5201/audit_v3_20261112`：候选缓存、近似候选 CSV、视觉抽查记录、编码分布与汇总。`split_config.json` 固定各清单指纹；加载时验证。

## 已知来源偏差

训练/开发/测试中的 Real 为 JPEG；AI 为 PNG。BigGAN 原始尺寸为 128×128，SD15 为 512×512，VQDM 为 256×256；Real 尺寸多样。统一模型输入尺寸不消除压缩与分辨率捷径。nature 的包名是镜像来源，不代表 VQDM 生成图片进入训练；其原始内容来源仍为 ImageNet。

当前随机抽样固定真假与生成器配额；没有验证跨生成器语义类别映射，不能宣称语义类别完全配平。报告须披露上述限制，后续用共同处理链检验失效，不能把 clean 性能当作通用检测能力。

## 训练选择与运行

A 使用 ImageNet 预训练 ResNet50，224 输入、batch64、10 epochs、AdamW lr=1e-4、weight_decay=1e-4、cosine scheduler、AMP、seed42、基础水平翻转。只用验证集 AUROC 选择 checkpoint，平分时用较低 BCE。

阈值取满足验证真实图经验 FPR≤1% 的最小阈值；边界相同分数向上越过，不拆分 ties。5,000 张验证真实图对应最多 50 次误报；这不保证测试或总体误报率为1%。阈值与训练状态保存在 checkpoint 中。未作概率校准。

当前 Docker data-root 为 `/var/lib/docker`，采用 containerd overlayfs snapshotter，containerd 默认数据位置为 `/var/lib/containerd`；BuildKit 系统目录也存在。有无关运行中容器，因此不停止/迁移服务，不创建 Docker 容器。使用 SSD 上的独立 `.venv-local` 解释器入口，复用 `/mnt/ssd/conda/project6/lib/python3.11/site-packages` 的现有依赖运行，所有依赖缓存、权重、日志与结果放 `/mnt/ssd`。实际环境为 Python 3.11.18、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124；完整版本见 `requirements-local.lock`。共享依赖只供加载，不执行升级或修改；后续重建可从锁定版本安装到独立环境。首次尝试在线安装旧版2.5.1因下载超时退出，未作系统级安装。

```bash
./scripts/run_local.sh test
./scripts/run_local.sh train --config configs/resnet50_A_v3.json \
  --manifest-dir data/manifests/v3 --name A_v3_s42
```

正式运行前先独立 smoke 吞吐测试；smoke checkpoint/指标不能进入正式结果。B/C 后续使用相同清单、seed、更新次数和双视图分布，仅 C 增加一致性项。
