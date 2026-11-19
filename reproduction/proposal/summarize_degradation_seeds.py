"""Aggregate 64k seen-validation predictions, pairing seeds and conditions."""
import json
from pathlib import Path
from statistics import mean, stdev

from reproduction.proposal.evaluate_degradations64k import CONDITIONS, summarize
from reproduction.proposal.run import sha, write
from reproduction.protocol import read_manifest

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
METHODS = ('frozen_L14_resize', 'effort', 'effort_degradation')


def read_json(path):
    return json.loads(path.read_text())


def statistics(values):
    return dict(mean=mean(values), sample_sd=stdev(values), by_seed=dict(zip(SEEDS, values)))


def verify_seed(seed, rows, manifest_hash):
    filename = 'validation_degradations64k.json' if seed == 42 else f'degradations_seed{seed}.json'
    config_path = HERE / filename
    config = read_json(config_path)
    assert config['seed'] == seed and config['conditions'] == CONDITIONS
    assert config['validation_sha256'] == manifest_hash and config['training_scale'] == 64000
    out = Path(config['output_root'])
    assert read_json(out/'status.json')['status'] == 'complete'
    scores = {}
    for spec in config['models']:
        method = spec['method']
        threshold_path = Path(spec['thresholds'])
        assert sha(threshold_path) == spec['thresholds_sha256']
        thresholds = read_json(threshold_path)['thresholds']
        scores[method] = {}
        for condition in CONDITIONS:
            receipt = read_json(out/method/f'{condition}.json')
            predictions = out/method/f'{condition}.csv'
            assert receipt['config_sha256'] == sha(config_path)
            assert receipt['prediction_sha256'] == sha(predictions)
            assert receipt['method'] == method and receipt['condition'] == condition
            assert receipt['status'] == 'complete' and not receipt['test_access']
            metrics = summarize(predictions, rows, thresholds, condition)
            assert metrics == receipt['metrics']
            fixed = metrics['p05']['overall']
            low = metrics['FPR01']['overall']
            scores[method][condition] = {
                'BA': fixed['balanced_accuracy'], 'AUROC': fixed['auroc'],
                'real_FPR': fixed['real_fpr'], 'low_FPR_recall': low['recall'],
                'low_FPR_real_FPR': low['real_fpr'],
                **{f'{g}_AUROC': v['auroc'] for g,v in metrics['p05']['by_generator'].items()}
            }
    assert set(scores) == set(METHODS)
    print(f'Verified seed {seed}: 270,000 predictions', flush=True)
    return scores


def main():
    ref = read_json(HERE/'validation_degradations64k.json')
    rows = read_manifest(ref['validation_manifest'])
    assert len(rows) == 10000
    manifest_hash = sha(ref['validation_manifest'])
    raw = {s: verify_seed(s, rows, manifest_hash) for s in SEEDS}
    methods = {}
    differences = {}
    metrics = raw[42]['effort'][CONDITIONS[0]].keys()
    for group in ['nine_condition_mean', *CONDITIONS]:
        conditions = CONDITIONS if group == 'nine_condition_mean' else [group]
        methods[group] = {}
        for method in METHODS:
            methods[group][method] = {
                metric: statistics([mean(raw[s][method][c][metric] for c in conditions) for s in SEEDS])
                for metric in metrics
            }
        differences[group] = {
            metric: statistics([
                mean(raw[s]['effort_degradation'][c][metric] - raw[s]['effort'][c][metric] for c in conditions)
                for s in SEEDS
            ]) for metric in metrics
        }
    report = dict(scope='64k seen validation; 10000 shared originals, 9 single corruptions',
                  units='fractions; BA and real_FPR at 0.5; low_FPR uses per-run clean FPR01 threshold',
                  seeds=SEEDS, models=methods, paired_augmented_minus_plain=differences)
    write(HERE/'main64k_three_seed_degradations.json', report)
    print(json.dumps(methods['nine_condition_mean'], indent=2))


if __name__ == '__main__':
    main()
