"""Freeze the proposal panel, or run the already frozen final evaluation."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import torch

from genimage_detector.metrics import choose_threshold, choose_fpr_threshold
from reproduction.proposal.evaluate_degradations_repeats import (
    CONDITIONS, corrupt, predict, summarize, verify_operations,
)
from reproduction.proposal.run import core, extra, data_loader, evaluate, sha, write
from reproduction.protocol import read_manifest, assert_disjoint

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'manifests/v1_20261114'
LOCK = HERE / 'P3_FROZEN.json'
OUT = Path('/mnt/ssd/data/logs/arin5201/reproduction/proposal_p3_20261119')
BASELINES = ['resnet50', 'deit_s', 'swin_t', 'cnnspot']
SOURCES = {'real': 5000, 'SD14': 2000, 'BigGAN': 2000, 'ADM': 2000, 'GLIDE': 2000}


def read(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Hash mismatch: {path}')


def panel():
    rows = []
    for source, count in SOURCES.items():
        name = 'test_' + source
        part = read_manifest(DATA / (name + '.csv'))
        assert len(part) == count
        assert all(r['split'] == name and r['source_split'] == 'val' and
                   r['generator'].lower() == source.lower() and
                   r['label'] == int(source != 'real') for r in part)
        rows.extend(part)
    return rows


class BaselineTransform:
    def __init__(self, method, meta, condition):
        self.transform = extra.Transform(method, False, meta)
        self.condition = condition

    def __call__(self, image, seed):
        return self.transform(corrupt(image, self.condition), seed)


def load(spec):
    method = spec['method']
    binding = core if method in core.CORE_METHODS else extra
    model, meta = binding.build(method)
    verify(spec['checkpoint'], spec['checkpoint_sha256'])
    classifier = None
    if meta['kind'] == 'features':
        classifier = pickle.loads(Path(spec['checkpoint']).read_bytes())
    else:
        state = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
    return model.cuda().eval(), meta, classifier


def score(spec, loaded, rows, condition):
    model, meta, clf = loaded
    method = spec['method']
    if method in core.CORE_METHODS:
        return predict(model, clf, meta, rows, method, condition)
    loader = data_loader(rows, BaselineTransform(method, meta, condition), spec['seed'])
    return evaluate(model, loader, meta)[0]


def freeze():
    if LOCK.exists():
        raise FileExistsError(LOCK)
    protocol = read(DATA / 'protocol.json')
    val = read_manifest(DATA / 'val.csv')
    test = panel()
    assert_disjoint(test, val, read_manifest(DATA / 'train_S2.csv'), read_manifest(DATA / 'dev.csv'))
    assert_disjoint(test, read_manifest(DATA / 'train_S1.csv'))
    files = {}
    for path in [Path(__file__), HERE/'evaluate_degradations_repeats.py',
                 HERE.parent/'protocol.py', HERE.parents[1]/'genimage_detector/metrics.py',
                 DATA/'protocol.json', HERE/'main64k_three_seed_degradations.json']:
        files[str(path.resolve())] = sha(path)
    for file in next(w for w in read(HERE.parent/'weights.lock.json')
                     if w['repo'] == 'openai/clip-vit-large-patch14')['files']:
        verify(file['path'], file['sha256'])
        files[file['path']] = file['sha256']

    specs = []
    for filename in ['scale64k.json', 'main64k_seed43.json', 'main64k_seed44.json', 'round1.json']:
        config_path = HERE / filename
        cfg = read(config_path)
        for p, h in cfg['implementation_sha256'].items():
            verify(p, h)
            files[p] = h
        files[str(config_path)] = sha(config_path)
        fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        methods = BASELINES if filename == 'round1.json' else core.CORE_METHODS
        for method in methods:
            folder = Path(cfg['output_root']) / method
            result = read(folder/'result.json')
            assert result['status'] == 'complete' and result['test_access'] is False
            assert (result['method'], result['seed'], result['config_sha256']) == (method, cfg['seed'], fingerprint)
            for path, digest in result['artifacts'].items():
                verify(path, digest)
            thresholds = read(folder/'thresholds.json')['thresholds']
            with (folder/'validation_scores.csv').open() as f:
                saved = list(csv.DictReader(f))
            assert [(r['original_id'], int(r['label'])) for r in saved] == [(r['original_id'], r['label']) for r in val]
            labels = [r['label'] for r in val]
            scores = np.array([float(r['score']) for r in saved])
            assert thresholds == dict(p05=.5, max_BA=choose_threshold(labels, scores),
                                      FPR01=choose_fpr_threshold(labels, scores))
            spec = dict(method=method, seed=cfg['seed'], training_config=str(config_path),
                        train_originals=result['train_originals'], best_epoch=result['detail'].get('best_epoch'),
                        thresholds=thresholds, checkpoint=result['checkpoint'],
                        checkpoint_sha256=result['artifacts'][result['checkpoint']],
                        training_artifacts=result['artifacts'], training_result_sha256=sha(folder/'result.json'))
            loaded = load(spec)
            replay = score(spec, loaded, val[:64], 'clean')
            np.testing.assert_allclose(replay, scores[:64], rtol=0, atol=1e-7)
            spec['validation_replay_max_error'] = float(abs(replay - scores[:64]).max())
            del loaded
            specs.append(spec)
            print('Verified', method, cfg['seed'], flush=True)

    means = read(HERE/'main64k_three_seed_degradations.json')['models']['nine_condition_mean']
    method = max(core.CORE_METHODS, key=lambda m: means[m]['BA']['mean'])
    demo = next(s for s in specs if s['method'] == method and s['seed'] == 42)
    write(LOCK, dict(models=specs, demo=dict(method=method, seed=42, threshold=demo['thresholds']['p05'],
               checkpoint=demo['checkpoint'], checkpoint_sha256=demo['checkpoint_sha256'],
               rule='Highest three-seed mean seen-validation nine-corruption BA at 0.5; use original seed 42',
               validation_mean_BA=means[method]['BA']['mean']), files=files,
               manifest_sha256={k:v for k,v in protocol['manifest_sha256'].items() if k in ['test_'+s for s in SOURCES]},
               originals_by_source=SOURCES, conditions=['clean', *CONDITIONS], expected_predictions=len(test)*10*len(specs),
               primary_endpoint='Equal-weight clean ADM and GLIDE AUROC, each against the shared real pool',
               comparisons=['effort - frozen_L14_resize', 'effort_degradation - effort'],
               checkpoint_selection=cfg['checkpoint_selection']))
    print('FROZEN', sha(LOCK), flush=True)


def run(digest):
    verify(LOCK, digest)
    cfg = read(LOCK)
    for p, h in cfg['files'].items():
        verify(p, h)
    for name, h in cfg['manifest_sha256'].items():
        verify(DATA/(name+'.csv'), h)
    rows = panel()
    for spec in cfg['models']:
        loaded = load(spec)
        method, seed = spec['method'], spec['seed']
        dest = OUT / f'{method}_seed{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        for condition in cfg['conditions']:
            threshold = spec['thresholds']
            path = dest / (condition + '.csv')
            receipt = dest / (condition + '.json')
            if receipt.exists():
                old = read(receipt)
                assert old['freeze_sha256'] == digest
                verify(path, old['prediction_sha256'])
                assert old['metrics'] == summarize(path, rows, threshold, condition)
                continue
            write(OUT/'status.json', dict(status='running', pid=os.getpid(), method=method,
                                          seed=seed, condition=condition, freeze_sha256=digest))
            print('START', method, seed, condition, flush=True)
            values = score(spec, loaded, rows, condition)
            tmp = path.with_suffix('.tmp')
            with tmp.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['original_id','source','condition','label','score', *['prediction_'+k for k in threshold]])
                for r, v in zip(rows, values, strict=True):
                    writer.writerow([r['original_id'], r['generator'], condition, r['label'], float(v),
                                     *[int(v >= t) for t in threshold.values()]])
            tmp.replace(path)
            metrics = summarize(path, rows, threshold, condition)
            write(receipt, dict(method=method, seed=seed, condition=condition, freeze_sha256=digest,
                                checkpoint_sha256=spec['checkpoint_sha256'], prediction_sha256=sha(path), metrics=metrics))
        del loaded
    write(OUT/'status.json', dict(status='complete', predictions=cfg['expected_predictions'], freeze_sha256=digest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--sha256')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    with (HERE/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        verify_operations()
        if args.action == 'freeze':
            freeze()
        else:
            try:
                run(args.sha256)
            except BaseException:
                write(OUT/'status.json', dict(status='failed', error=traceback.format_exc()))
                raise
