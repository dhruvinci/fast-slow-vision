"""Four-class posture probe on frozen, mean-pooled tower features (E07, E08).

The same head for every encoder: standardise and PCA to 64 dimensions on training rows only, then a
source-balanced ridge regression to one-hot classes (lambda = 1, 64 x 4 + 4 = 260 supervised parameters).
Scores are uncalibrated ridge outputs, not probabilities. Whole source families are held out together.

Rows are dicts with 'family', 'role' ('train' / 'development' / 'audit'), 'known' (bool) and 'label' (class index).

  python3 src/posture_probe.py features.npz rows.json      # prints per-fold metrics

Environment: Python 3.10+, numpy.
"""
import argparse
import collections
import json

import numpy as np

CLASSES = ['upright', 'crouched_or_kneeling', 'seated_or_reclined', 'lying']
PCA_DIM = 64


def make_folds(rows):
    """One fold per training family (that family held out), then development and audit against all training."""
    train = [i for i, r in enumerate(rows) if r['role'] == 'train' and r['known']]
    folds = []
    for family in sorted({rows[i]['family'] for i in train}):
        folds.append({'name': 'holdout:' + family, 'train': [i for i in train if rows[i]['family'] != family],
                      'test': [i for i in train if rows[i]['family'] == family]})
    for role in ('development', 'audit'):
        test = [i for i, r in enumerate(rows) if r['role'] == role and r['known']]
        if test:
            folds.append({'name': role, 'train': train, 'test': test})
    for f in folds:
        assert not {rows[i]['family'] for i in f['train']} & {rows[i]['family'] for i in f['test']}
    return folds


def fit(x, y, families, ridge=1.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or not np.isfinite(x).all() or min(x.shape) <= PCA_DIM:
        raise ValueError('need finite features with more than 64 rows and dimensions')
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    z = (x - mean) / scale
    _, _, v = np.linalg.svd(z, full_matrices=False)
    components = v[:PCA_DIM]
    z = z @ components.T / 8
    counts = collections.Counter(families)
    weights = np.array([1 / counts[f] for f in families])
    weights *= len(weights) / weights.sum()
    target = np.eye(len(CLASSES))[y]
    xm = np.average(z, axis=0, weights=weights)
    ym = np.average(target, axis=0, weights=weights)
    zx = (z - xm) * np.sqrt(weights[:, None])
    zy = (target - ym) * np.sqrt(weights[:, None])
    coef = np.linalg.solve(zx.T @ zx + ridge * np.eye(PCA_DIM), zx.T @ zy)
    return {'mean': mean, 'scale': scale, 'components': components, 'coef': coef, 'intercept': ym - xm @ coef}


def predict(model, x):
    z = ((np.asarray(x, dtype=float) - model['mean']) / model['scale']) @ model['components'].T / 8
    return z @ model['coef'] + model['intercept']


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(y, pred):
        conf[a, b] += 1
    recalls = [conf[c, c] / conf[c].sum() if conf[c].sum() else None for c in range(len(CLASSES))]
    present = [r for r in recalls if r is not None]
    return {'n': int(len(y)), 'correct': int((y == pred).sum()), 'accuracy': float((y == pred).mean()),
            'macro_recall_present_classes': float(np.mean(present)), 'confusion': conf.tolist(),
            'recall': dict(zip(CLASSES, recalls))}


def run(features, rows):
    out = {}
    for f in make_folds(rows):
        model = fit(features[f['train']], [rows[i]['label'] for i in f['train']],
                    [rows[i]['family'] for i in f['train']])
        pred = predict(model, features[f['test']]).argmax(1)
        out[f['name']] = metrics([rows[i]['label'] for i in f['test']], pred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features', help='npz with a "vectors" array (src/encoder_features.py)')
    ap.add_argument('rows', help='json list of row dicts aligned with the vectors')
    a = ap.parse_args()
    x = np.load(a.features)['vectors'].astype(np.float64)
    rows = json.load(open(a.rows))
    print(json.dumps(run(x, rows), indent=2))


if __name__ == '__main__':
    main()
