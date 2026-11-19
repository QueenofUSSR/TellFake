import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path('results/final_20261113');D=json.loads((R/'results.json').read_text());C=D['conditions'];M=['A','B','C'];COL=['#555555','#2366a8','#c46a22'];LABEL=['clean','JPEG50','resize .25','blur 1','resize→JPEG','JPEG→resize','repeat JPEG','strong chain']
rows=list(csv.DictReader((R/'per_seed_metrics.csv').open()));train=list(csv.DictReader((R/'training.csv').open()))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
fig,axes=plt.subplots(1,2,figsize=(10,3.2))
for method,color in zip(M,COL):
 for metric,ax in [('train_loss',axes[0]),('val_loss',axes[1])]:
  y=[np.mean([float(r[metric]) for r in train if r['method']==method and int(r['epoch'])==e]) for e in range(1,11)]
  ax.plot(range(1,11),y,label=method,color=color);ax.set_xlabel('Epoch');ax.set_ylabel('BCE / total objective');ax.set_yscale('log');ax.legend()
axes[0].set_title('Training objective (seed mean)');axes[1].set_title('Clean validation BCE (seed mean)');fig.tight_layout();fig.savefig(R/'training.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for method,color in zip(M,COL):
 for ax,metric in zip(axes,['balanced_accuracy','recall']):
  mean=[];std=[]
  for c in C:
   values=[np.mean([float(r[metric]) for r in rows if r['method']==method and int(r['seed'])==s and r['condition']==c and r['group'] in (['BigGAN','SD15','VQDM','Midjourney'] if metric=='balanced_accuracy' else ['VQDM','Midjourney'])]) for s in [42,43,44]]
   mean.append(np.mean(values)*100);std.append(np.std(values,ddof=1)*100)
  ax.errorbar(range(8),mean,yerr=std,label=method,color=color,marker='o',markersize=3,capsize=2)
  ax.set_xticks(range(8),LABEL,rotation=45,ha='right');ax.set_ylabel('Percent (mean ± seed SD)');ax.legend()
axes[0].set_title('Macro balanced accuracy: four generators');axes[1].set_title('Unknown AI recall: VQDM + Midjourney');fig.tight_layout();fig.savefig(R/'robustness.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for ax,group in zip(axes,['ImageNet','COCO']):
 for method,color in zip(M,COL):
  mean=[];std=[]
  for c in C:
   values=[100*float(r['real_fpr']) for r in rows if r['method']==method and r['group']==group and r['condition']==c];mean.append(np.mean(values));std.append(np.std(values,ddof=1))
  ax.errorbar(range(8),mean,yerr=std,label=method,color=color,marker='o',markersize=3,capsize=2)
 ax.axhline(1,color='black',linestyle=':',linewidth=.8,label='Validation target 1%');ax.set_title(group+' actual test false-positive rate');ax.set_xticks(range(8),LABEL,rotation=45,ha='right');ax.set_ylabel('FPR % (mean ± seed SD)');ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(R/'false_positives.png');plt.close(fig)
intervals=D['paired_intervals'];fig,ax=plt.subplots(figsize=(8.8,3.2))
a=[next(r for r in intervals if r['scope']==c and r['metric']=='macro_balanced_accuracy') for c in C]
y=np.array([r['delta'] for r in a])*100;lo=np.array([r['low95'] for r in a])*100;hi=np.array([r['high95'] for r in a])*100
ax.errorbar(range(8),y,yerr=np.stack([y-lo,hi-y]),fmt='o',capsize=3,color=COL[2]);ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(8),LABEL,rotation=35,ha='right');ax.set_ylabel('C − B balanced accuracy (percentage points)');ax.set_title('Paired original-image bootstrap: pointwise 95% CI');fig.tight_layout();fig.savefig(R/'paired.png');plt.close(fig)
print('Four report figures generated')
