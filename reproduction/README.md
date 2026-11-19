# P0 数据与环境准备

更新：2026-11-14。**核心数据与环境闭环通过验收，可以进入 P1/S0；未启动实验训练。** 机器可读验收见 `P0_ACCEPTANCE.json`。P0 通过不表示所有原方法实验的数据均已齐全，也不表示方法级 S0、吞吐或训练结果已经完成。

## 已冻结的数据

盘点现有保留库存 894,000 张，全部路径存在。本轮对 302,976 张候选完整解码，并补解码旧 v2 的 14,000 张图片；旧 v3/补充数据使用已有像素与 dHash 记录。对全部 56,995 张历史已用图片执行排除，不只排除旧最终测试。候选坏图 0 张；额外保守排除 5,713 张。旧 SD14 的 3 张、ADM 的 6 张坏图仍在原 quarantine，不恢复、不删除。

U 轨道的 seen 为 SD14+BigGAN，primary unseen 为 ADM+GLIDE。真实/假图为 1:1，假图在两 seen 来源均分；按来源/标签与可识别的文件名类别前缀分层，不虚构未知类别：

- S0/S1/S2/S3：4,000 / 16,000 / 64,000 / 128,000 张，严格嵌套。
- validation：10,000；dev：8,000，均来自官方 train，与所有训练规模独立。
- 确认测试：SD14、BigGAN、ADM、GLIDE、SD15、VQDM、Midjourney 各 2,000 张，共用 5,000 张真实图，共 19,000 张独立原图；均来自官方 val，排除历史已用原图。共享 real 不重复计数。
- 另有两个各 16,000 张的 R 专用训练池。连同 U 各分区，共冻结 197,000 张独立原图；嵌套训练清单不重复计数。
- R_SD14、R_BigGAN、R_BigGAN_ADM 派生了 S0/S1/S2 方法级清单，与 U 共用已审计原图，不能跨轨道当作独立数据重复计数。

入口：`manifests/v1_20261114/protocol.json`；每份 CSV 有 SHA-256，逐图包含来源、官方 split、原图/分组 ID、文件/像素哈希、尺寸、格式、dHash、JPEG subsampling、量化表摘要和可匹配的质量估计。不匹配标准 IJG 量化表时不猜测 JPEG quality。

独立验收检查嵌套性、数量、身份/文件/像素/近重复泄漏、旧清单排除及解码后文件未变化。使用 EXIF 归一化 RGB SHA-256、ImageNet 文件名身份和 dHash≤4 连通分组；每组最多保留一个代表。视觉抽查发现低细节图片的 dHash 误报，因此明确按保守排除处理，不声称候选都是真重复，不保证发现裁剪或全部感知重复。确认测试仍来自历史研究过的数据域。

本轮完整像素审计只覆盖明确列出的候选，**不是对全部 894,000 张库存的像素认证**。后续纳入未选库存必须另行审计。旧 v3/补充清单与实验产物保持原样；没有删除数据、模型、报告或 Docker 数据。

## 环境与模型

独立环境位于 SSD 的 `env-core` 与 `env-iapl`；Python 3.11.18、torch 2.6.0+cu124、transformers 4.46.3，完整传递依赖见 `envs/*.lock`。均通过依赖一致性检查，CUDA 可用；没有升级 `.venv-local`。与 mixinran 自报新版环境的差异已记录，不能称为原环境的精确复刻。

CLIP L/14、B/16、B/32、SD2.1 VAE 和 IAPL 所需 OpenAI 原版 L/14 JIT 均固定 revision/上游 SHA-256，完成下载校验及实际加载，见 `weights.lock.json` 和 `environment_validation.json`。OpenAI CLIP MIT 授权文本及 SD2.1 openrail++ 模型卡随权重保留。IAPL 独立环境的原生模型构建另行检查。

五个上游仓库 commit/工作区保持不变，三份补丁在原 commit 上 `git apply --check` 通过。补丁覆盖验证集边界、禁止缺失来源回退、双分支累积尾批归一化、IAPL 固定模型路径/验证清单、VAE 固定路径与逐父图 seed。方法差异及后续边界见 `METHOD_DIFFERENCES.md`。

## 当前命令

在项目根目录运行：

```bash
bash reproduction/run.sh test
bash reproduction/run.sh validate-data
bash reproduction/run.sh reproduction/scripts/prepare_run.py --method dual_adapter
```

最后一条仅生成 S0 配置，不启动训练。新版本已验收后，`prepare_data.py` 会拒绝覆盖该冻结版本。训练、续跑、特征缓存、逐图评分器及完整方法接口的端到端验收属于 P1/P2/P3，不能将本轮合成层测试算作正式 S0。

## 明确未完成的扩展项

RAISE/Mirage/EvalGEN/Wukong 在检查的本地目录内未找到，登记于 `data_provenance.json`。因此完整 D3j/D3c、作者八源 GenImage 均值与相关外部测试仍有数据缺项；不得用简化数据冒充。GAPL 的 Community Forensics 两阶段方案仍为可选项。

ADM 的本地来源、CRC 解包、坏图隔离和保留库存可追溯；原归档在本任务开始前已被移除，未找到可复查的外部归档 SHA-256 凭证。不能把本轮本地文件哈希解释为重新完成了外部归档真实性认证。

实验矩阵登记了 209 个待办/候选项，**不是已批准的正式训练队列**；S3 面板、epoch/update 配方、GPU 小时和磁盘预算须根据 P1 每方法至少 100 个有效 batch 的实测冻结。IAPL TTA/OIS 的逐图重置与单 GPU 覆盖尚未验收，因此训练入口暂不开放原生 TTA 路径。DeiT/Swin/CNNSpot 等附加骨干权重按 P1 任务再获取。

## 证据位置

- 审计、排除明细、近重复样例与日志：`/mnt/ssd/data/logs/arin5201/reproduction/p0_20261114/`
- 环境、固定源码、权重与缓存：`/mnt/ssd/data/cache/arin5201/reproduction/`
- 数据验收：`data_validation.json`；方法差异：`METHOD_DIFFERENCES.md`；来源与缺项：`data_provenance.json`。
- 本轮专用测试 10 项、旧测试 25 项均通过。专用测试覆盖清单篡改、泄漏、派生身份、缺失源报错、真实清单加载、尾批归一化和合成 Adapter/SSF/Effort/LoRA 层；不外推为完整训练通过。
