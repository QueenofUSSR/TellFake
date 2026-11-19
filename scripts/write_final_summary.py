import json
from pathlib import Path
R=Path('results/final_20261113');d=json.loads((R/'results.json').read_text())
def m(method,metric,scope='non_strong'):
 return next(r for r in d['overall'] if r['method']==method and r['metric']==metric and r['scope']==scope)
def ci(metric):return next(r for r in d['paired_intervals'] if r['scope']=='non_strong' and r['metric']==metric)
p=ci('macro_balanced_accuracy');f=ci('COCO_FPR')
lines=['# 最终实验结果（2026-11-13）','', 'A/B/C各三个seed、共九次正式训练及全部八条件最终评估已完成。B/seed44原失败运行保留，同批次AMP重算修复后完整重跑；其他seed未剔除。','',
 f"主要比较为B/C。七种非强退化条件下，C相对B的平均宏Balanced Accuracy变化为 {100*p['delta']:+.2f} 个百分点，原图配对bootstrap 95%区间为 [{100*p['low95']:+.2f}, {100*p['high95']:+.2f}]。区间{'包含零，不能据此确认稳定提升' if p['low95']<=0<=p['high95'] else '不含零，但只反映这三个模型条件下的图像抽样不确定性'}。",'', '所有百分数为三个seed的均值±样本标准差；先在每个seed内对七种非强条件平均。']
for method in ['A','B','C']:
 vals=[m(method,k) for k in ['macro_balanced_accuracy','unknown_recall','ImageNet_FPR','COCO_FPR']]
 lines.append('- '+method+'：'+ '；'.join(f"{label} {100*v['mean']:.2f}% ± {100*v['std']:.2f}%" for label,v in zip(['宏BA','未知生成器召回','ImageNet误报','COCO误报'],vals)) )
lines+=['','强退化条件单独报告：']
for method in ['A','B','C']:
 a=m(method,'macro_balanced_accuracy','strong');b=m(method,'COCO_FPR','strong');lines.append(f"- {method}：宏BA {100*a['mean']:.2f}%；COCO误报 {100*b['mean']:.2f}%。")
lines+=['',f"已核验 {d['prediction_files']} 个逐图CSV、{d['prediction_rows_verified']:,} 条预测记录；独立最终原图15,995张。共享的5,000张ImageNet真实图仅计一次，COCO4,995张单独报告。所有阈值来自验证集，测试中未重新拟合。",'', '置信区间按原图分组并保留所有处理版本及匹配B/C预测，1000次分层bootstrap。先平均三个seed，再重采样原图，因此不是训练随机性与图像不确定性的联合区间；训练差异另以seed SD报告。各区间未做多重比较校正。','', '限制：PNG/JPEG与分辨率偏差、有限生成器覆盖、可能的预训练数据接触、近似查重不完备、COCO非逐图相机真实性认证、模拟处理链而非实平台测试。不能宣称通用图片真实性检测或测试误报恒为1%。','', '六页报告：`output/pdf/arin5201_report_20261113.pdf`。可重算结果及置信区间：`results/final_20261113/`。原始预测路径及SHA-256见 `prediction_ledger.csv`；九次模型登记见日志目录 `final_20261113/verified_runs.json`。', '', '未执行范围扩容或新架构实验。']
Path('docs/FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
p=Path('README.md');s=p.read_text();s='''当前九次正式训练、八条件最终评估、统计核验和六页报告已完成。见 [最终结果](docs/FINAL_RESULTS.md)。原始训练队列的失败及恢复经过保留在执行记录和日志中。

'''+s;p.write_text(s)
p=Path('docs/PROGRESS.md');s=p.read_text();s='''# 最终交付完成（2026-11-13）

九次正式训练已完成，包括修复后的B/C seed44。全部最终预测已按冻结协议生成，逐图核验与统计汇总通过。最终数字及边界见 [FINAL_RESULTS](FINAL_RESULTS.md)，六页PDF与图表由验证过的CSV生成。下面保留执行过程中的历史状态，不代表当前仍在训练。

'''+s;p.write_text(s)
q=Path('/mnt/ssd/data/logs/arin5201/final_20261113/status.json');q.write_text(json.dumps(dict(stage='complete',formal_runs=9,final_evaluations=9,statistics_verified=True,report='output/pdf/arin5201_report_20261113.pdf'),indent=2))
print('\n'.join(lines[:8]))
