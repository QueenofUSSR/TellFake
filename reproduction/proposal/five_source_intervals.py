"""Paired clean macro-AUROC contrasts, conditional on the three frozen training seeds."""
import csv
from pathlib import Path

import numpy as np
from reproduction.proposal.extend_p3 import OLD, NEW, OUT, REPORT, table
from reproduction.proposal.p3 import read

METHODS = ('frozen_L14_resize', 'effort', 'effort_degradation')
SOURCES = ('Real', *OLD, *NEW)


def ranks(scores, labels):
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_scores))]
    group = np.cumsum(np.isin(np.arange(len(order)), starts)) - 1
    return order, starts, group, labels[order]


def auc(weights, layout):
    order, starts, group, positive = layout
    w = weights[order]
    negatives = np.add.reduceat(w * (~positive), starts)
    lower_and_half_ties = np.cumsum(negatives) - negatives * 0.5
    return np.dot((w * positive), lower_and_half_ties[group]) / (10000 * 5000)


def main():
    assert read(OUT/'status.json')['status'] == 'complete'
    ids = None
    layouts = []
    for method in METHODS:
        for seed in (42, 43, 44):
            with (OUT / f'{method}_seed{seed}' / 'clean.csv').open() as f:
                rows = list(csv.DictReader(f))
            keys = [(r['original_id'], r['source'], int(r['label'])) for r in rows]
            if ids is None:
                ids = keys
            assert ids == keys
            labels = np.array([r[2] for r in keys], dtype=bool)
            layouts.append(ranks(np.array([float(r['score']) for r in rows]), labels))

    sources = np.array([r[1] for r in ids])
    groups = [np.flatnonzero(sources == name) for name in SOURCES]
    assert [len(group) for group in groups] == [5000] + [2000]*5
    rng = np.random.default_rng(20261119)
    records = []
    weights = np.ones(len(ids), dtype=np.int32)
    # Equal fake source sizes make pooled AUROC exactly equal to source-macro AUROC.
    points = np.array([auc(weights, layout) for layout in layouts]).reshape(3, 3)
    for method, values in zip(METHODS, points, strict=True):
        for seed, value in zip((42,43,44), values, strict=True):
            stats = read(OUT / f'{method}_seed{seed}' / 'clean.json')['metrics']['p05']['by_generator']
            np.testing.assert_allclose(value, np.mean([m['auroc'] for m in stats.values()]), rtol=0, atol=1e-12)
    draws = []
    for _ in range(1000):
        for group in groups:
            weights[group] = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
        estimates = np.array([auc(weights, layout) for layout in layouts]).reshape(3, 3)
        draws.append(np.diff(estimates.mean(axis=1)))
    for idx, deltas in enumerate(np.diff(points, axis=0)):
        lo, hi = np.percentile(np.asarray(draws)[:, idx], [2.5, 97.5])
        records.append(dict(comparison=METHODS[idx+1]+' minus '+METHODS[idx], metric='clean_macro5_auroc',
                            seed42=deltas[0], seed43=deltas[1], seed44=deltas[2], mean=deltas.mean(),
                            seed_sd=deltas.std(ddof=1), image_bootstrap_low=lo, image_bootstrap_high=hi,
                            draws=1000, rng_seed=20261119))
    table(REPORT / 'clean_auroc_contrasts.csv', records)


if __name__ == '__main__':
    main()
