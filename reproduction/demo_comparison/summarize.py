"""Summarize paired predictions; bootstrap originals within each source."""
import csv
import numpy as np
from reproduction.demo_comparison.compare import OUT, CONDITIONS, sha, write
import json

NAMES = ['frozen_L14_resize','effort_degradation','thomas_b16_jpeg','dual_adapter']
sources = ['Real','SD14','BigGAN','ADM','GLIDE']


def main():
    ids = json.loads((OUT / "protocol.json").read_text())["ids"]
    stats = []
    by_image = []
    for name in NAMES:
        correct = []
        condition_stats = []
        for condition in CONDITIONS:
            path = OUT/name/(condition+'.csv')
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert receipt['prediction_sha256'] == sha(path)
            with path.open() as f:
                rows = list(csv.DictReader(f))
            assert [r['original_id'] for r in rows] == ids
            assert [r['source'] for r in rows] == ['Real']*600 + sum(([g]*300 for g in sources[1:]),[])
            y = np.array([int(r['label']) for r in rows])
            p = np.array([float(r['score']) for r in rows])
            ok = ((p >= .5) == y)
            correct.append(ok)
            g = receipt['metrics']['by_generator']
            stats_row = dict(accuracy=ok.mean(), common_unseen_tpr=ok[1200:].mean(),
                             adm_tpr=g['ADM']['recall'],glide_tpr=g['GLIDE']['recall'],
                             real_fpr=1-ok[:600].mean(),
                             common_unseen_auc=np.mean([g[k]['auroc'] for k in ('ADM','GLIDE')]))
            condition_stats.append(stats_row)
            stats.append(dict(method=name,condition=condition,**stats_row))
        stats.append(dict(method=name,condition='degraded_mean',
                          **{k:np.mean([r[k] for r in condition_stats[1:]]) for k in condition_stats[0]}))
        by_image.append(np.array(correct))
    # Paired image-level intervals for all candidates; each original keeps its ten conditions.
    groups = [np.arange(600),*[np.arange(i,i+300) for i in range(600,1800,300)]]
    rng = np.random.default_rng(20261119)
    contrasts = []
    a = np.asarray(by_image,dtype=float)
    for label,values in [('clean',a[:,0,:]),('degraded_mean',a[:,1:,:].mean(axis=1))]:
        draws = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(group,size=len(group)) for group in groups])
            draws.append(values[:,idx].mean(axis=1))
        draws = np.array(draws)
        for i in range(len(NAMES)):
            for j in range(i):
                diff = draws[:,i]-draws[:,j]
                contrasts.append(dict(condition=label,method=NAMES[i],reference=NAMES[j],
                                      difference=float(values[i].mean()-values[j].mean()),
                                      ci95=np.quantile(diff,[.025,.975]).tolist()))
    write(OUT/'summary.json',dict(metrics=stats,paired_accuracy=contrasts))
    for row in stats:
        if row['condition'] in ['clean','degraded_mean']:
            print(row['method'],row['condition'],{k:round(float(v)*100,2) for k,v in row.items() if k not in ['method','condition']})

if __name__ == '__main__':
    main()
