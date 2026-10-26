"""Matched spatial ownership decoder: which actor (A1, A2) or background owns each image cell (E08).

Inputs per frame: a 64-channel 28x28 feature grid (train-only PCA of one tower's tokens, or 32 + 32 channels from
two towers) and two appearance queries (mean feature inside a manually chosen reference box of each actor on
frame 0 of the clip). The decoder is shared across actors, so swapping the queries swaps the A1/A2 outputs exactly
(by construction, not by learning). 24,738 trainable parameters for every representation. Each frame is decoded on
its own: no temporal memory, no pose, no anatomy.

Training uses only sparse reviewed labels (0 background, 1 actor A1, 2 actor A2, 3 unknown) pooled to 56x56 cells,
ignoring cells below 90% purity, with a class-balanced loss. Study recipe: AdamW lr 1e-3, weight decay 0.01,
batch 6, 120 epochs, gradient clip 1, seeds 7 / 17 / 29, mean logits of the three seeds; one source clip held out
per fold.

Environment: torch >= 2.4, numpy.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, channels=64, hidden=32):
        super().__init__()
        self.visual = nn.Conv2d(channels, hidden, 3, padding=1)
        self.query = nn.Linear(channels, hidden)
        self.relation = nn.Sequential(nn.Conv2d(4 * hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        self.background = nn.Conv2d(hidden, 1, 1)

    def forward(self, x, q):
        """x: B x C x 28 x 28 features; q: B x 2 x C queries. Returns B x 3 x 56 x 56 logits (bg, A1, A2)."""
        v = F.gelu(self.visual(x))
        qs = F.gelu(self.query(q))
        scores = [self.background(v)]
        for actor in range(2):
            p = qs[:, actor, :, None, None].expand_as(v)
            scores.append(self.relation(torch.cat((v, p, v * p, (v - p).abs()), 1)))
        return F.interpolate(torch.cat(scores, 1), size=(56, 56), mode='bilinear', align_corners=False)


def sparse_counts(labels):
    """448x448 label image (0 bg, 1 A1, 2 A2, >=3 unknown) -> 3 x 56 x 56 class counts, impure cells zeroed."""
    a = torch.as_tensor(np.asarray(labels).copy())
    assert a.shape == (448, 448)
    counts = torch.stack([(a == k).float().reshape(56, 8, 56, 8).sum((1, 3)) for k in range(3)])
    purity = counts.max(0).values / counts.sum(0).clamp_min(1)
    return counts * (purity >= .9).unsqueeze(0)


def balanced_loss(logits, counts):
    losses = -(F.log_softmax(logits, dim=1) * counts).sum((0, 2, 3))
    n = counts.sum((0, 2, 3))
    keep = n > 0
    assert keep.any(), 'no reviewed pixels in this batch'
    return (losses[keep] / n[keep]).mean()


def fit_pca(training_tokens, k=64):
    """Standardise + PCA on training tokens only; SVD signs fixed so refits replay exactly."""
    assert training_tokens.ndim == 2 and len(training_tokens) > k
    mean = training_tokens.mean(0)
    scale = training_tokens.std(0).clip(1e-5)
    _, s, vt = np.linalg.svd((training_tokens - mean) / scale, full_matrices=False)
    basis = vt[:k].T.copy()
    for j in range(k):
        if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
            basis[:, j] *= -1
    return {'mean': mean, 'scale': scale, 'basis': basis}


def project(grid, p):
    """28 x 28 x D grid -> 64 x 28 x 28 features."""
    return (((grid.astype(np.float32) - p['mean']) / p['scale']) @ p['basis']).transpose(2, 0, 1)


def appearance_queries(features, refs):
    """features: C x 28 x 28; refs: two normalised xyxy boxes on the clip's first frame (never masks)."""
    centres = (np.arange(28) + .5) / 28
    vectors = []
    for x0, y0, x1, y1 in refs:
        mask = (centres[:, None] >= y0) & (centres[:, None] < y1) & (centres[None, :] >= x0) & (centres[None, :] < x1)
        assert mask.any(), 'reference must contain at least one patch centre'
        vectors.append(features[:, mask].mean(1))
    return np.stack(vectors).astype(np.float32)


def metrics(predictions, labels, contacts):
    """Sparse scores: actor IoU and recall over labelled pixels only; contact ownership over reviewed contact pixels.
    Unknown pixels are excluded, so spill into unlabelled background is not penalised (not a full-mask IoU)."""
    cm = np.zeros((3, 3), dtype=np.int64)
    correct = total = 0
    for pred, lab, con in zip(predictions, labels, contacts):
        valid = lab < 3
        cm += np.bincount(lab[valid].astype(int) * 3 + pred[valid], minlength=9).reshape(3, 3)
        known = con > 0
        correct += int((pred[known] == con[known]).sum())
        total += int(known.sum())
    tp = cm.diagonal()
    den = cm.sum(1) + cm.sum(0) - tp
    iou = np.divide(tp, den, out=np.zeros(3), where=den > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    return {'actor_iou': float(iou[1:].mean()), 'macro_recall': float(recall[cm.sum(1) > 0].mean()),
            'contact_owner_accuracy': correct / total if total else None, 'confusion': cm.tolist()}


def train_one(x, q, counts, seed, epochs=120, batch=6, device='cpu'):
    """x: N x 64 x 28 x 28, q: N x 2 x 64, counts: N x 3 x 56 x 56 (training frames of the other clips)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Decoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for k in range(0, len(x), batch):
            idx = torch.as_tensor(order[k:k + batch], device=device)
            opt.zero_grad(set_to_none=True)
            loss = balanced_loss(model(x[idx], q[idx]), counts[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
    return model.eval()


if __name__ == '__main__':
    print('trainable parameters:', sum(p.numel() for p in Decoder().parameters()))
