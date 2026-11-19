"""Evaluate three reserved generators and combine them with the frozen P3 predictions."""
import csv
import fcntl
import json
from pathlib import Path
from statistics import mean, stdev
import traceback

import torch
from reproduction.proposal import p3
from reproduction.protocol import assert_disjoint, read_manifest

DIGEST = 'f8c1464090e517c8faa2046e95dd3b2c9be738ae49fddb3e233c1c2d76d68b51'
NEW = ('SD15', 'VQDM', 'Midjourney_subset')
OLD = ('ADM', 'GLIDE')
OUT = p3.OUT.with_name('proposal_p3_five_unseen_20261119')
REPORT = p3.HERE.parents[1] / 'results/p3_five_unseen'
METRICS = ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auroc', 'average_precision', 'real_fpr')


def table(path, records):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    tmp.replace(path)


def evaluate():
    p3.verify(p3.LOCK, DIGEST)
    cfg = p3.read(p3.LOCK)
    for path, digest in cfg['files'].items():
        p3.verify(path, digest)
    old_panel = p3.panel()
    old_rows = [r for r in old_panel if r['generator'] in ('Real', *OLD)]
    new_rows = []
    for source in NEW:
        part = read_manifest(p3.DATA / f'test_{source}.csv')
        assert len(part) == 2000 and all(r['label'] == 1 and r['source_split'] == 'val'
            and r['generator'] == source and r['split'] == 'test_' + source for r in part)
        new_rows.extend(part)
    assert_disjoint(new_rows, old_panel, *(read_manifest(p3.DATA / (s + '.csv')) for s in ('train_S2', 'val', 'dev')))
    assert_disjoint(new_rows, read_manifest(p3.DATA / 'train_S1.csv'))
    rows = old_rows + new_rows
    assert len(rows) == 15000
    protocol = dict(p3_sha256=DIGEST, manifest_sha256={s:p3.sha(p3.DATA / f'test_{s}.csv')
                    for s in ('real', *OLD, *NEW)}, conditions=cfg['conditions'],
                    reused=['Real', *OLD], new=list(NEW), thresholds='original per-model P3 thresholds')
    p3.write(OUT/'protocol.json', protocol)
    details = []
    for spec in cfg['models']:
        model = spec['method']
        seed = spec['seed']
        dest = OUT / f'{model}_seed{seed}'
        dest.mkdir(exist_ok=True)
        loaded = None
        thresholds = spec['thresholds']
        for condition in cfg['conditions']:
            path = dest / (condition + '.csv')
            receipt = path.with_suffix('.json')
            base = p3.OUT / dest.name / path.name
            ref = p3.read(base.with_suffix('.json'))
            assert ref['freeze_sha256'] == DIGEST and ref['checkpoint_sha256'] == spec['checkpoint_sha256']
            p3.verify(base, ref['prediction_sha256'])
            if receipt.exists():
                old = p3.read(receipt)
                assert old['protocol'] == protocol and old['p3_prediction_sha256'] == ref['prediction_sha256']
                p3.verify(path, old['prediction_sha256'])
            else:
                p3.write(OUT/'status.json', dict(status='running', method=model, seed=seed, condition=condition))
                print('START', model, seed, condition, flush=True)
                if loaded is None:
                    loaded = p3.load(spec)
                values = p3.score(spec, loaded, new_rows, condition)
                with base.open() as f:
                    records = [r for r in csv.DictReader(f) if r['source'] in ('Real', *OLD)]
                assert len(records) == len(old_rows)
                for r, score in zip(new_rows, values, strict=True):
                    records.append(dict(original_id=r['original_id'], source=r['generator'], condition=condition,
                        label=1, score=float(score), **{'prediction_'+k:int(score >= t) for k,t in thresholds.items()}))
                table(path, records)
            metrics = p3.summarize(path, rows, thresholds, condition)
            for key, result in metrics.items():
                for source in OLD:
                    assert result['by_generator'][source] == ref['metrics'][key]['by_generator'][source]
                for source, m in result['by_generator'].items():
                    details.append(dict(method=model, seed=seed, condition=condition,
                                        threshold_rule=key, source=source, **m))
            p3.write(receipt, dict(protocol=protocol, checkpoint_sha256=spec['checkpoint_sha256'],
                prediction_sha256=p3.sha(path), p3_prediction_sha256=ref['prediction_sha256'], metrics=metrics))
        del loaded
    REPORT.mkdir(parents=True, exist_ok=True)
    table(REPORT/'by_source.csv', details)
    summary = []
    for model in dict.fromkeys(r['method'] for r in details):
        for rule in ('p05', 'max_BA', 'FPR01'):
            for clean in (True, False):
                subset = [r for r in details if r['method']==model and r['threshold_rule']==rule
                          and (r['condition']=='clean')==clean]
                for source in ('macro5', *OLD, *NEW):
                    group = subset if source == 'macro5' else [r for r in subset if r['source']==source]
                    seeds = sorted({r['seed'] for r in group})
                    item = dict(method=model, threshold_rule=rule, conditions='clean' if clean else 'nine_degradations',
                                source=source, seeds=len(seeds))
                    for metric in METRICS:
                        means = [mean(r[metric] for r in group if r['seed']==s) for s in seeds]
                        item[metric] = mean(means)
                        item[metric+'_seed_sd'] = stdev(means) if len(means)>1 else None
                    summary.append(item)
    table(REPORT/'summary.csv', summary)
    p3.write(OUT/'status.json', dict(status='complete', new_predictions=780000,
             reused_predictions=1170000, model_conditions=130, source_metric_rows=len(details),
             summaries={s:p3.sha(REPORT/s) for s in ('by_source.csv','summary.csv')}))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with (p3.HERE/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            p3.verify_operations()
            evaluate()
        except BaseException:
            p3.write(OUT/'status.json', dict(status='failed', error=traceback.format_exc()))
            raise
