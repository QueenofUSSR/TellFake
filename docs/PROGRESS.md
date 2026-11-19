> 历史记录：本文件描述旧 ResNet/v3 实验，过程中的“待运行”等状态仅代表当时情况。当前工作以 `plan.md` 和 `reproduction/README.md` 为准。

# 最终交付完成（2026-11-13）

九次正式训练已完成，包括修复后的B/C seed44。全部最终预测已按冻结协议生成，逐图核验与统计汇总通过。最终数字及边界见 [FINAL_RESULTS](FINAL_RESULTS.md)，六页PDF与图表由验证过的CSV生成。下面保留执行过程中的历史状态，不代表当前仍在训练。

# 最终评估跟进（2026-11-13，进行中）

用户已授权训练后直接进行最终评估、统计和报告。原正式队列完成7次后，在B/seed44第7个epoch因AMP非有限梯度停止，失败记录保留。新增同批次AMP重算：降低loss scale、回滚BatchNorm与RNG、不丢弃更新，25项测试通过。B44从同一seed完整重跑，前六个epoch的训练/验证loss与原运行完全一致；C44采用同一修复。成功的其他7次运行继续保留。

当前流水线状态见 `/mnt/ssd/data/logs/arin5201/final_20261113/status.json`，错误见同目录 `failure.json`（如存在）。它会先核验九次训练、阈值、验证预测与源码指纹，再按冻结八条件评估四个测试池；不重新选择lambda或阈值。统计与报告脚本已准备，待预测全部通过核验后运行。

# 执行记录：2026-11-13

已补充并冻结Midjourney 2,000张与COCO外部真实4,995张；基础v3及A权重不变，实验总量为42,995张。原COCO5,000张均完整解码，5张近似重复候选保守排除。补充来源、独立指纹、归档校验、许可与标签限制见 [SUPPLEMENT_V3](SUPPLEMENT_V3.md)。两个新测试集仍未运行模型预测。

## B/C 开发筛选已完成

预先固定协议在 `/mnt/ssd/data/logs/arin5201/development_20261113/screening_protocol.json`。B、C(lambda0.1)、C(lambda1)均使用16,000张训练图、7,000张验证图、seed42、2epochs及500次更新；开发评估为4,000张图片的五个条件。三组均成功，所有逐图预测和指标已经核验重算。

五条件平均Balanced Accuracy：B为95.56%，C0.1为95.86%，C1为95.73%；平均实际开发真实FPR分别为1.32%、1.16%、1.25%。C0.1相对B提高0.30个百分点BA，FPR降低0.16个百分点，按预定规则选为候选。clean BA未下降，均通过预定误报及clean表现约束。

这只是单seed、2epochs开发信号，尚无最终泛化或统计显著性结论。每个条件都对同一批图片配对比较；不把五个处理版本视为独立图片。记录为 `decision.json`、`conditions.csv` 和 `paired_changes.csv`；没有依据最终测试调整权重。24项协议测试通过，包括新增补充清单隔离和真实单类FPR评估。

## 正式协议已冻结，串行训练队列已启动

固定C lambda=0.1、seed42/43/44、10epochs、每次2,500次更新。复用已完成的A_v3_s42，其数据、运行环境、超参数、A变换均一致；旧版本缺少后来加入的梯度记录，源码差异保留并披露。其他8次训练逐一执行，失败即停止后续步骤，保留日志。

队列目录 `/mnt/ssd/data/logs/arin5201/formal_20261113`：`protocol.json` 固定清单、配置、主要比较、阈值与最终处理条件；`source/` 为本次独立冻结源码和数据清单，后续工作区编辑不影响队列；`status.json` 记录当前运行、完成列表及退出状态。全程使用SSD本地环境，不创建Docker容器。按试跑吞吐估计剩余约45分钟，实际以状态为准。

最终八条件固定为clean、JPEG50、缩小恢复0.25、blur1、resize0.5→JPEG50、JPEG50→resize0.5、JPEG70→JPEG50、resize0.25→JPEG50→blur1。最后一条作为强退化单独报告，均为模拟处理链。阈值在各运行自己的验证集拟合后跨来源和条件固定。主要比较B/C，保留全部seed；后续报告均值/标准差、逐来源及最差组，样本置信区间按原图分组。COCO只单独报告真实FPR。

队列只训练，不自动打开最终测试。结束后核验全部运行，再进入独立最终评估。2epoch pilot与之前smoke均不计入正式10epoch结果。

---

# 执行记录：2026-11-12

本轮已依次完成候选审计、v3 划分、吞吐与保存/加载检查、A 正式训练、B/C 实现和 GPU 联调。最终测试没有运行。

## 数据与审计

`data/manifests/v3` 固定 seed42 和六个清单指纹，共36,000张图片：train16,000、val7,000、dev4,000、seen_test2,000、unseen_test2,000、real_test5,000。test 中真假分别保存，评估入口按已见/未知 AI 配同一真实池。

审计候选50,000张，完整解码成功；保守排除旧v2内容，检查文件/像素哈希、ImageNet原图身份及近似候选分组。选中36,000张的路径、字节、像素、原图身份与组件均不跨集合，选中图片间无 dHash 汉明距离≤4的候选。近似方法不能排除所有派生图。

审计目录：`/mnt/ssd/data/logs/arin5201/audit_v3_20261112/`。关键产物：`audit_summary.json`、`selected_verification.json`、`distribution_summary.json`、`visual_review.json`。

仍缺 Midjourney 和5,000张独立外部真实照片。PNG/JPEG 与原始分辨率存在显著来源偏差，必须在正式结论中披露。v3是缩减的自定义协议，不是原计划全部完成。详见 `DATA_V3.md`。

## A 正式训练

运行名 `A_v3_s42`，2026-11-12 16:39–16:42（香港时间）完成，退出码0。使用全部16,000张训练图及7,000张验证图，10 epochs、2,500次更新，`smoke_test=false`。

- 最佳权重：`/mnt/ssd/data/ckpt/arin5201/A_v3_s42/best.pt`
- 最后权重：同目录 `last.pt`
- 日志、配置、环境、完整源码快照、清单和验证预测：`/mnt/ssd/data/logs/arin5201/A_v3_s42/`
- 进程退出状态及控制台：`/mnt/ssd/data/logs/arin5201/launches/A_v3_s42/`

模型按验证AUROC选出epoch10。训练及验证平均每epoch16.76秒，峰值CUDA分配显存3.06 GiB；这不是整张GPU的总显存。实际运行环境与旧基线不同，使用PyTorch2.6.0+cu124；本轮B/C也固定同一环境。

验证阈值0.6730842590332032，对5,000张验证真实图经验误报率1%（50次误报）。这仅是验证拟合结果，不是最终测试性能或总体误报保证。`completion_verification.json` 已确认7,000条验证预测与清单一致、指标和阈值可由CSV重算、源码快照与清单指纹一致。

## B/C 实现与验证

B和C均从同一原图独立采样两个视图：每个视图独立水平翻转，再均匀选择clean、JPEG70、缩小恢复0.5、缩放→JPEG、JPEG→缩放；最后统一224输入及归一化。JPEG固定4:2:0，缩放固定bicubic。

每次更新拼接两个视图执行同一次前向，分类损失为两个视图BCE的平均。C另外增加lambda乘以对称Bernoulli Jensen–Shannon损失，采用float32对数域计算，两侧均接收梯度。候选非零权重仅0.1与1.0；B权重0。两组相同样本、seed、shuffle、随机视图分布与更新次数。

新增组合处理链入口、开发集评估入口、非有限梯度检查、分类/一致性分项日志和吞吐/显存记录。配置为 `configs/resnet50_B_v3.json`、`configs/resnet50_C_js01_v3.json`、`configs/resnet50_C_js1_v3.json`。

21项协议测试通过，包括官方val隔离、同源泄漏拒绝、验证阈值ties、JS对称性/边界/有限梯度、lambda0值及梯度等价、处理顺序与确定性。GPU smoke分别运行B、C0、C0.1，两epochs、512张训练图、256张验证图；B/C0训练与验证记录相同，最佳验证预测CSV逐字节一致。C0.1损失和梯度有限，预测非恒定。双视图smoke峰值CUDA分配显存约5.74 GiB。

C0.1的开发集评估smoke覆盖全部五种条件，每条件128张，确认全部路径属于dev且没有混入最终真实测试池。checkpoint及逐图预测保留在 `smoke_B_v3_s42`、`smoke_C0_v3_s42`、`smoke_C01_v3_s42`、`smoke_C01_dev_v3_s42` 对应目录。所有这些smoke指标均不能用于方法去留或正式结果。

```bash
./scripts/run_local.sh test
./scripts/run_local.sh train --config configs/resnet50_B_v3.json \
  --manifest-dir data/manifests/v3 --name B_v3_pilot_s42 --epochs 2
./scripts/run_local.sh train --config configs/resnet50_C_js01_v3.json \
  --manifest-dir data/manifests/v3 --name C_js01_v3_pilot_s42 --epochs 2
./scripts/run_local.sh evaluate --manifest-dir data/manifests/v3 \
  --checkpoint /mnt/ssd/data/ckpt/arin5201/B_v3_pilot_s42/best.pt \
  --splits dev --name B_v3_pilot_dev_s42 \
  --conditions clean jpeg:70 resize:0.5 'resize:0.5>jpeg:70' 'jpeg:70>resize:0.5'
```

上述pilot命令是下一步入口，尚未执行。先在相同短训练预算下筛选B/C，并用dev的真假表现决定候选；随后冻结正式多seed队列。A已完成的正式运行可计入A组seed42，B/C试跑不能因名称相似就计为正式10epoch结果。

## 保留的失败与限制

首次旧版PyTorch依赖下载超时，改用SSD现成依赖，不修改Docker存储或其他运行中容器。`.venv-local` 使用新Python3.11解释器加载现有依赖目录，版本在 `requirements-local.lock` 固定；不要升级共享依赖。依赖目录是运行前提，重建环境需按锁安装。

首次 `smoke_A_v3_throughput_s42` 因新增v3字段与旧CSV导出不兼容失败，保留目录与日志；修复后以 `_r2` 新名称通过。A训练出现部分来源图片EXIF元数据警告，但像素完整解码成功，未据此删除或重标标签。

没有进行Docker构建、创建或迁移。数据准备只增加清单、审计日志和缓存，不修改原始图片。当前工程无Git仓库，复核依据是运行时源文件快照与SHA-256。
