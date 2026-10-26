"""Hip-relative movement of limb-pair midpoints between two frames, and the selective-answer policy (E05).

For wrists and for ankles separately: average the left and right joint, subtract the hip midpoint, take the
change between the two frames (eight source frames apart in the study) and divide by the image diagonal. Each
axis is then 'decreased' (< -0.02), 'little change' or 'increased' (> +0.02), boundaries inclusive in the middle.
Pooling left and right removes the question of which side is which (the named-joint version failed partly on
that); the price is that opposite movements of the two limbs cancel.

Points use COCO-17 indexing (as returned by RTMPose / COCO keypoints): wrists 9, 10; ankles 15, 16; hips 11, 12.

The policy keeps the largest complete confidence-tie group that reaches >= 50% coverage with <= 15% error on a
calibration set, or accepts nothing; it is then frozen before the confirmation set is opened.

Environment: Python 3.10+ (standard library only).
"""
import math

FAMILIES = {'wrists': (9, 10), 'ankles': (15, 16)}
HIPS = (11, 12)
DEADBAND = 0.02


def classify(value, deadband=DEADBAND):
    if not math.isfinite(value):
        raise ValueError('class value must be finite')
    return 0 if value < -deadband else 2 if value > deadband else 1   # decreased / little change / increased


def movement(points, present, family, width, height, hips=HIPS):
    """points[t][j] = (x, y) or None, present[t][j] = bool for t in (0, 1). Returns delta (dx, dy) or None."""
    if len(points) != 2 or len(present) != 2:
        raise ValueError('two frames required')
    joints = tuple(FAMILIES[family]) + tuple(hips)
    relative = []
    for t in range(2):
        if not all(present[t][j] and points[t][j] is not None for j in joints):
            return None
        mid = [(points[t][FAMILIES[family][0]][a] + points[t][FAMILIES[family][1]][a]) / 2 for a in range(2)]
        root = [(points[t][hips[0]][a] + points[t][hips[1]][a]) / 2 for a in range(2)]
        relative.append([mid[a] - root[a] for a in range(2)])
    diagonal = math.hypot(width, height)
    return [(relative[1][a] - relative[0][a]) / diagonal for a in range(2)]


def family_confidence(scores, family, hips=HIPS):
    """Rank for the policy: the minimum score of the eight required points (not a probability)."""
    joints = tuple(FAMILIES[family]) + tuple(hips)
    return max(0., min(1., min(scores[t][j] for t in range(2) for j in joints)))


def select_policy(rows, minimum_coverage=.5, maximum_error=.15):
    """rows: dicts with returned (bool), correct (bool), confidence (float or None). Missing rows count against
    coverage. Returns the frozen threshold, or an accept-nothing policy when no threshold qualifies."""
    total = len(rows)
    candidates = []
    for threshold in sorted({r['confidence'] for r in rows if r['returned']}, reverse=True):
        accepted = [r for r in rows if r['returned'] and r['confidence'] >= threshold]
        coverage = len(accepted) / total
        error = sum(not r['correct'] for r in accepted) / len(accepted)
        if coverage >= minimum_coverage and error <= maximum_error:
            candidates.append((len(accepted), threshold, coverage, error))
    if not candidates:
        return {'feasible': False, 'threshold': None, 'coverage': 0., 'error': None}
    n, t, c, e = max(candidates, key=lambda x: (x[0], -x[1]))
    return {'feasible': True, 'threshold': t, 'coverage': c, 'error': e, 'accepted': n}


def accepted(row, policy):
    return bool(policy['feasible'] and row['returned'] and row['confidence'] >= policy['threshold'])


def same_support_errors(rows):
    """Mean absolute axis error of the method and of 'no movement' on exactly the rows the method returned."""
    got = [r for r in rows if r['returned']]
    if not got:
        return None
    mae = sum(abs(r['predicted'] - r['true']) for r in got) / len(got)
    zero = sum(abs(r['true']) for r in got) / len(got)
    return {'returned': len(got), 'valid': len(rows), 'axis_mae': mae, 'zero_axis_mae': zero,
            'relative_reduction': 1 - mae / zero if zero else None}
