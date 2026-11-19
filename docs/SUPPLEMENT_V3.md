> 历史记录：本文件描述旧 ResNet/v3 实验，过程中的“待运行”等状态仅代表当时情况。当前工作以 `plan.md` 和 `reproduction/README.md` 为准。

> 最终状态（2026-11-13）：九次正式训练、288组最终评估、统计重算及六页报告已完成。详情见 [最终结果](FINAL_RESULTS.md)。下文训练队列和待评估描述为历史记录。

# 独立测试补充协议：2026-11-13

清单目录 `data/manifests/v3_supplement_20261113`。基础v3的六个清单及指纹保持不变；补充配置记录对应基础指纹，并独立固定两个新清单的SHA-256。现有A checkpoint继续有效。

## 来源与数量

Midjourney位于 `/mnt/ssd/data/datasets/GenImage/official_v3/Midjourney_subset`。26批传输清单记录21,549张图片，其中原始val/ai 6,000张，train/ai 15,549张。本次只对全部6,000张val做完整解码、文件SHA-256与批次传输哈希比对、像素哈希和dHash审计，随机固定其中2,000张独立图片进入 `midjourney_test.csv`。所有Midjourney库存都保留未知生成器测试身份，不进入拟合或开发。

外部目录 `/mnt/ssd/data/datasets/val2017` 有5,000张JPEG，完整解码成功。保留的本地归档 `/mnt/ssd/data/downloads/arin5201/coco/val2017.zip.part` 虽仍为.part后缀，但ZIP目录完整，5,000个本地文件的长度与CRC全部匹配归档。记录归档自行计算的SHA-256，不冒充官方校验。

[COCO官方下载页](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)列出val2017为5K图片，提供原始图片包入口。图片版权与注释许可不同，图片使用遵守原始Flickr条款，不能把注释的CC BY4.0当作全部图片统一许可；见[官方使用条款](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)。本项目只保存清单及本机路径，不重新分发来源图片。没有取得逐张Flickr许可/来源元数据，因此不声称逐图许可认证或相机原片认证。

本次所有11,000张候选完成解码；对照基础v3全部36,000张及旧v2的精确内容查重，并在新候选间与v3间检索dHash汉明距离≤4。共12张近似候选保守排除：Midjourney 7张、COCO 5张；没有将它们认定为已确认重复或标签错误。最终外部真实池为 **4,995张**，不复制或补回被排除图片凑5,000。

对接受的COCO照片按seed42固定随机抽查32张缩略图，所见为普通摄影场景，含摆拍/影棚画面；这不是逐图相机真实性认证。采用COCO自然/非生成图片代理标签，普通修图允许为Real；可能含拼贴、绘画或其他传统编辑，报告必须披露。

## 隔离与评估

两个补充集合均为最终测试专用，未运行任何模型预测，不用于选择checkpoint、阈值、一致性权重或处理链。每张记录路径、标签、来源、原始划分、尺寸、格式、文件/像素哈希、原图身份、分组与dHash。基础清单与补充清单不共享已知精确内容或检索到的近似候选。近似检索不保证检出所有裁剪/派生图，且不覆盖整个储备库。

Midjourney与原v3的共享5,000张真实测试池配对，真实图片不重复累计为独立样本。COCO4,995张只单独报告真实误报率及样本数；二分类召回、AUROC等不适用，不把它与GenImage真实池混合后掩盖来源差异。所有条件使用checkpoint保存的验证阈值。

审计目录 `/mnt/ssd/data/logs/arin5201/supplement_20261113` 保存所有候选、排除记录、像素/格式分布、归档校验及抽查记录。数据根目录为 `/mnt/ssd/data/datasets`，补充CSV路径相对于此目录。

入口已经实现并用合成数据测试：基础指纹不匹配、补充清单改动、越界路径和跨集合重复均被拒绝；真实单类评估可以输出FPR而不产生NaN或重拟合阈值。

以下为最终冻结方法后使用的入口，本轮尚未执行：

```bash
./scripts/run_local.sh evaluate --manifest-dir data/manifests/v3 \
  --supplement-dir data/manifests/v3_supplement_20261113 \
  --checkpoint /mnt/ssd/data/ckpt/arin5201/A_v3_s42/best.pt \
  --splits midjourney_test external_real_test \
  --name A_v3_s42_supplement_final --conditions clean
```

最终应使用冻结的完整处理条件队列，不因这个clean命令示例就提前查看最终测试。
