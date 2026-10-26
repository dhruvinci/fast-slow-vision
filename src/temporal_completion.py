"""Pose-space temporal completion: a small adapter that learns a residual over linear interpolation (E06).

Input: nine poses (12 torso/limb joints, Penn Action order = COCO joints 5..16) with availability flags; the
middle pose is to be filled in. The model is a 3-layer 1-D temporal convolution (68,280 parameters) whose output
layer starts at zero, so step 0 *is* linear interpolation. Averaging over a left/right side swap makes it
equivariant to exchanging sides (it does not resolve anatomical naming). No RGB, no Laya features.

Training (study recipe): synthetic hiding of one middle frame, three middle frames, or the middle wrists/ankles;
10% context dropout; 0.005-diagonal observation noise (a follow-up set it to 0); AdamW lr 1e-3, wd 1e-4,
batch 64, 800 updates, smooth-L1; validation every 100 steps, earliest minimum wins, step 0 included. In both
runs step 0 won, so the selected model was plain interpolation.

  python3 src/temporal_completion.py windows.npz        # npz: poses N x 9 x 12 x 2, visible N x 9 x 12,
                                                         # times N x 9, role N (train/validation/...)

Penn Action labels (x, y, visibility per frame, 13 joints) can be turned into windows with `source_arrays`.
Environment: torch >= 2.4, numpy (scipy only to read Penn Action .mat files).
"""
import argparse
import math

import numpy as np
import torch
from torch import nn

SIDE_SWAP = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
DISTAL = [4, 5, 10, 11]            # wrists and ankles within the 12 joints
CONDITIONS = ('frame_gap', 'three_frame_gap', 'distal_gap')
PENN_JOINTS = np.arange(1, 13)     # drop Penn's head point (it differs from COCO's nose)


def source_arrays(x, y, visibility, frames_one_based, width, height):
    """Penn Action coordinates -> diagonal-normalised (9, 12, 2) poses, (9, 12) validity, (9,) times."""
    x, y, visibility = map(np.asarray, (x, y, visibility))
    idx = np.asarray(frames_one_based) - 1
    raw = np.stack((x[idx][:, PENN_JOINTS], y[idx][:, PENN_JOINTS]), -1).astype(np.float64)
    ok = visibility[idx][:, PENN_JOINTS].astype(bool) & np.isfinite(raw).all(-1)
    ok &= (raw[..., 0] >= .5) & (raw[..., 0] <= width + .5) & (raw[..., 1] >= .5) & (raw[..., 1] <= height + .5)
    poses = np.zeros((len(idx), 12, 2), np.float32)
    poses[ok] = ((raw[ok] - 1.0) / math.hypot(width, height)).astype(np.float32)
    return poses, ok, np.asarray(frames_one_based, dtype=np.float64)


def prepare_inputs(poses, visible, times):
    """Centre the observed joints, add availability flags, and compute the interpolation baseline for frame 4."""
    p = np.asarray(poses, dtype=np.float32)
    v = np.asarray(visible, dtype=bool)
    t = np.asarray(times, dtype=np.float64)
    if p.shape != (9, 12, 2) or v.shape != (9, 12) or not (np.diff(t) > 0).all():
        raise ValueError('expected nine twelve-joint poses with increasing times')
    clean = np.where(v[..., None], p, 0)
    origin = clean.sum((0, 1)) / max(int(v.sum()), 1)
    x = np.concatenate([np.where(v[..., None], clean - origin, 0), v[..., None].astype(np.float32)], -1)
    baseline = np.zeros((12, 2), np.float32)
    past = baseline.copy()
    supported = np.zeros(12, bool)
    for j in range(12):
        before = np.flatnonzero(v[:4, j])
        after = np.flatnonzero(v[5:, j]) + 5
        if v[4, j]:
            baseline[j] = past[j] = p[4, j]
            supported[j] = True
        elif len(before) and len(after):
            a, b = before[-1], after[0]
            w = (t[4] - t[a]) / (t[b] - t[a])
            baseline[j] = p[a, j] * (1 - w) + p[b, j] * w
            past[j] = p[a, j]
            supported[j] = True
    return dict(input=x.astype(np.float32), baseline=baseline, past=past, supported=supported)


def corrupt(poses, visible, condition, rng=None, training=False, noise=.005, dropout=.1):
    """Hide the evaluation targets; in training also drop 10% of context and add observation noise."""
    p = np.asarray(poses, dtype=np.float32).copy()
    v = np.asarray(visible, dtype=bool).copy()
    hidden = np.zeros(12, bool)
    if condition in ('frame_gap', 'three_frame_gap'):
        hidden[:] = True
    elif condition == 'distal_gap':
        hidden[DISTAL] = True
    else:
        raise ValueError(condition)
    for i in ([3, 4, 5] if condition == 'three_frame_gap' else [4]):
        v[i, hidden] = False
    if training:
        v &= rng.random(v.shape) >= dropout
        p += rng.normal(0, 1, p.shape).astype(np.float32) * noise
    p[~v] = 0
    return p, v, hidden & np.asarray(visible, dtype=bool)[4]


class TemporalCompletion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(36, 96, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(96, 96, 3, padding=2, dilation=2), nn.GELU(),
                                     nn.Conv1d(96, 96, 3, padding=1), nn.GELU())
        self.output = nn.Linear(96, 24)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def residual(self, x):
        z = x.reshape(x.shape[0], 9, 36).transpose(1, 2)
        return self.output(self.encoder(z)[:, :, 4]).reshape(x.shape[0], 12, 2)

    def forward(self, x, baseline):
        r = self.residual(x)
        s = self.residual(x[:, :, SIDE_SWAP])[:, SIDE_SWAP]
        return baseline + (r + s) * .5


def make_batch(poses, visible, times, indices, conditions, rng=None, training=False, noise=.005):
    keys = ('input', 'baseline', 'past', 'supported')
    out = {k: [] for k in keys + ('target', 'target_mask')}
    for i, c in zip(indices, conditions):
        p, v, mask = corrupt(poses[i], visible[i], c, rng, training, noise)
        d = prepare_inputs(p, v, times[i])
        for k in keys:
            out[k].append(d[k])
        out['target'].append(poses[i, 4])
        out['target_mask'].append(mask)
    return {k: np.stack(v) for k, v in out.items()}


def masked_mse(model, batches, device='cpu'):
    num = count = 0.
    with torch.no_grad():
        for b in batches.values():
            pred = model(torch.from_numpy(b['input']).to(device), torch.from_numpy(b['baseline']).to(device))
            m = b['target_mask'] & b['supported']
            num += float(np.square(pred.cpu().numpy() - b['target'])[m].sum())
            count += int(m.sum()) * 2
    return num / count


def train(poses, visible, times, train_idx, val_idx, steps=800, noise=.005, seed=24601, device='cpu'):
    """Fit and select the earliest validation minimum, including step 0 (= interpolation)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TemporalCompletion().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    val = {c: make_batch(poses, visible, times, val_idx, [c] * len(val_idx)) for c in CONDITIONS}
    model.eval()
    best, best_step = masked_mse(model, val, device), 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = [(0, best)]
    for step in range(1, steps + 1):
        idx = rng.choice(train_idx, 64, replace=True)
        conds = [CONDITIONS[c] for c in rng.integers(0, len(CONDITIONS), 64)]
        b = make_batch(poses, visible, times, idx, conds, rng, True, noise)
        mask = torch.from_numpy(b['target_mask'] & b['supported']).to(device)
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(torch.from_numpy(b['input']).to(device), torch.from_numpy(b['baseline']).to(device))
        loss = nn.functional.smooth_l1_loss(pred[mask], torch.from_numpy(b['target']).to(device)[mask], beta=.01)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        if step % 100 == 0:
            model.eval()
            v = masked_mse(model, val, device)
            history.append((step, v))
            if v < best:
                best, best_step = v, step
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best_step, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('windows')
    ap.add_argument('--noise', type=float, default=.005)
    a = ap.parse_args()
    d = np.load(a.windows, allow_pickle=False)
    role = d['role']
    model, step, history = train(d['poses'], d['visible'], d['times'], np.flatnonzero(role == 'train'),
                                 np.flatnonzero(role == 'validation'), noise=a.noise)
    print('parameters', sum(p.numel() for p in model.parameters()), 'selected step', step)
    for s, v in history:
        print(f'step {s:4d}  validation MSE {v:.3e}')


if __name__ == '__main__':
    main()
