"""Grounding controls for multiple-choice decision models (E01, E02).

A control pairs an original question with a changed version and states how the answer should change:
  - option reorder:   same meaning, so the same answer (consistency);
  - actor swap:       A and B exchange names, so the answer maps A <-> B (equivariance);
  - blank image:      the evidence is gone, so a grounded model should change its answer or say it can't tell;
  - time reversal:    'increased' <-> 'decreased'; scored on directional references only, both answers must be
                      right (all-case consistency can pass vacuously: 'unchanged' and 'insufficient evidence'
                      map to themselves).
Changed-pixel controls measure sensitivity only; they never invent a correct answer for the changed input.

Also: averaging a question's raw option logits over option orders and over A/B renaming, which removes position
and letter bias by construction (it does not show the model learned to ignore them).

Environment: Python 3.10+ (standard library only).
"""
import itertools
import math


def top(probabilities):
    """Deterministic argmax over a {option: probability} dict (ties broken by sorted option id)."""
    return max(sorted(probabilities), key=probabilities.get)


def consistency(pairs, mapping=None):
    """pairs: [(original_probs, changed_probs)]. Share whose changed answer equals mapping(original answer)."""
    mapping = mapping or {}
    ok = [mapping.get(top(a), top(a)) == top(b) for a, b in pairs]
    return sum(ok), len(ok)


def retention(pairs, references=None):
    """Blank / shuffled image: how many top answers stay the same (and how many previously correct ones)."""
    same = [top(a) == top(b) for a, b in pairs]
    out = {'unchanged': sum(same), 'total': len(pairs)}
    if references is not None:
        correct = [top(a) == r for (a, _), r in zip(pairs, references)]
        out['correct_before'] = sum(correct)
        out['correct_kept'] = sum(s and c for s, c in zip(same, correct))
    return out


def directional_both_correct(pairs, references, flip=(('increased', 'decreased'), ('decreased', 'increased'))):
    """Time reversal on directional references only: original and reversed answers must both be right."""
    m = dict(flip)
    rows = [(a, b, r) for (a, b), r in zip(pairs, references) if r in m]
    return sum(top(a) == r and top(b) == m[r] for a, b, r in rows), len(rows)


def all_orders(options):
    """Every ordering of the answer options (6 for A / B / unknown)."""
    return [list(p) for p in itertools.permutations(options)]


def average_logits(runs):
    """runs: list of {option: raw logit} from different option orders (already keyed by canonical option id).
    Returns the mean logit per option; apply temperature/softmax afterwards."""
    keys = runs[0].keys()
    return {k: sum(r[k] for r in runs) / len(runs) for k in keys}


def identity_average(original_runs, renamed_runs, swap=(('A', 'B'), ('B', 'A'))):
    """Average raw logits over option orders AND over A/B renaming (renamed answers mapped back first)."""
    m = dict(swap)
    mapped = [{m.get(k, k): v for k, v in r.items()} for r in renamed_runs]
    return average_logits(list(original_runs) + mapped)


def softmax(logits, temperature=1.0):
    z = {k: v / temperature for k, v in logits.items()}
    mx = max(z.values())
    e = {k: math.exp(v - mx) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}
