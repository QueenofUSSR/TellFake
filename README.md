# TellFake

ARIN5201 课程项目，使用 GenImage 研究跨生成器 AI 图像检测。最终评估包含 Frozen CLIP、Effort、Effort＋退化增强与四个补充基线。

- [中文实验报告（PDF）](output/reports/GenImage_实验报告_20261119.pdf) · [可编辑版本](output/reports/GenImage_实验报告_20261119.docx)
- [五源评估、指标口径与复算说明](results/p3_five_unseen/README.md)
- [模型权重与校验文件](https://github.com/QueenofUSSR/TellFake/releases/tag/p3-20261119) · [模型文件清单](release_models.json)

## 运行单图检测

使用 Python 3.12。先安装适合本机的 PyTorch 2.6.0 与 torchvision 0.21.0，再安装 transformers 4.46.3、scikit-learn 1.9.1、NumPy 和 Pillow。原实验环境的详细版本见 [core.lock](reproduction/envs/core.lock)。从仓库根目录运行：

```bash
python scripts/download_release_models.py
python -m reproduction.demo.serve --weights release_assets --port 7860
```

浏览器打开 http://127.0.0.1:7860/，选择 JPEG 或 PNG。默认下载约 1.71 GB，包含冻结的 CLIP ViT-L/14 与 seed 42 分类器。运行 `python scripts/download_release_models.py --all` 可下载全部 13 个正式模型及共享骨干（约 10.16 GB）。每个文件均按 SHA-256 校验。只加载来自可信来源的 pickle 或 PyTorch checkpoint。

## 实验复核

训练使用 SD1.4 和 BigGAN；未见来源包括 ADM、GLIDE、SD1.5、VQDM 和 Midjourney。权重清单按方法与 seed 区分，原始 checkpoint 的选择规则、阈值、哈希见 [P3_FROZEN.json](reproduction/proposal/P3_FROZEN.json)。训练和完整评测脚本沿用原机绝对路径，迁移环境时需按清单准备 GenImage 数据、依赖上游源码及路径映射。仓库包含数据划分清单，不分发 GenImage 原图。预训练骨干来自 openai/clip-vit-large-patch14，版本和来源见 [weights.lock.json](reproduction/weights.lock.json)。

## 研究记录

### AI 图像检测：复现与规模化比较

当前主计划：[plan.md](plan.md)。当前准备状态与入口：[P0 准备记录](reproduction/README.md)。

2026-11-14：SD1.4 与 ADM 已进入本地库存；P0 核心数据与环境准备已完成并通过验收，扩展数据缺项单独登记。P1/S0 已完成 24 个方法/配置验收与吞吐测量，见 [P1 报告](reproduction/p1/README.md)。S1/S2 配方与预算已冻结，尚未开始正式训练或最终测试。

## 历史实验（已归档）

ResNet50 A/B/C 三 seed、九次训练及八条件评估已完成。历史结果见 [最终结果](docs/FINAL_RESULTS.md)，历史实施过程见 [实施计划](IMPLEMENTATION_PLAN.md) 和 [执行记录](docs/PROGRESS.md)。这些结果不代表新 CLIP/Effort 复现已完成。

旧 v3 与补充清单保持不变；其最终测试已被查看，只用于历史回归。新实验不将旧测试原图或其派生内容用于训练。

原始数据、权重、环境、日志和缓存存放于 `/mnt/ssd`。本轮使用独立 Python 环境；Docker 存储迁移不在本轮操作范围内。

### 单图检测 demo

运行 `bash reproduction/run.sh reproduction/demo/serve.py`，打开 <http://127.0.0.1:7860/> 并选择 JPEG 或 PNG。该展示使用 P3 后横评选出的本地 Frozen CLIP ViT-L/14（64k、seed 42）及其 logistic regression 分类器，固定阈值 0.5。上传图片仅在内存中处理。
