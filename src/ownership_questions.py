"""Limb-ownership questions from COCO person keypoints, and the geometry baselines they are scored against (E02).

A question shows a visible wrist or ankle and two people, each marked by a torso reference box (the shoulder/hip
rectangle shrunk by 20% per side). Answer: which person owns the limb. Rules, as in the study:
  - only photos licensed CC BY 2.0 or CC BY-SA 2.0 on Flickr (COCO license ids 4 and 5);
  - all four torso keypoints labelled visible (v = 2), reference box at least 16 px on its short side;
  - target keypoint visible, inside the owner's box and polygon mask (3 px tolerance), no other person's visible
    wrist/ankle within 6 px, and a target box (2.5% of the long side) that touches neither torso box;
  - each torso box at least 85% inside its own person's mask and at most 10% inside the other's.
The reference is COCO membership: a published annotation, not an independently adjudicated ownership judgement.

Geometry baselines (no Laya features):
  - nearest_torso: squared distance from the target point to each torso box (0 inside); ties -> unknown;
  - nearest_tool_point: nearest predicted wrist/ankle of the same family from a pose tool (unknown if missing).

  python3 src/ownership_questions.py $FSV_DATA/coco/annotations/person_keypoints_train2017.json out.jsonl

Environment: Python 3.10+, numpy.
"""
import argparse
import json
import math

import numpy as np

JOINTS = {9: 'left_wrist', 10: 'right_wrist', 15: 'left_ankle', 16: 'right_ankle'}
LICENSES = (4, 5)


def reference_box(ann):
    if ann.get('iscrowd') or len(ann.get('keypoints', [])) != 51:
        return None
    p = [ann['keypoints'][3 * i:3 * i + 3] for i in (5, 6, 11, 12)]
    if any(v != 2 for _, _, v in p):
        return None
    xs, ys = [x for x, _, _ in p], [y for _, y, _ in p]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    box = [x0 + .2 * w, y0 + .2 * h, x1 - .2 * w, y1 - .2 * h]
    return box if min(box[2] - box[0], box[3] - box[1]) >= 16 else None


def polygon_mask(polygons, width, height):
    """Even-odd rasterisation at pixel centres (not byte-identical to the COCO rasteriser)."""
    if not isinstance(polygons, list) or not polygons:
        return None
    out = np.zeros((height, width), dtype=bool)
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2:
            return None
        p = np.array(poly, dtype=float).reshape(-1, 2)
        x0, x1 = max(0, int(math.floor(p[:, 0].min()))), min(width, int(math.ceil(p[:, 0].max())))
        y0, y1 = max(0, int(math.floor(p[:, 1].min()))), min(height, int(math.ceil(p[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        x = np.arange(x0, x1)[None, :] + .5
        y = np.arange(y0, y1)[:, None] + .5
        inside = np.zeros((y1 - y0, x1 - x0), bool)
        for (ax, ay), (bx, by) in zip(p, np.roll(p, -1, axis=0)):
            if ay != by:
                inside ^= ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / (by - ay) + ax)
        out[y0:y1, x0:x1] |= inside
    return out


def _intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _in_xywh(box, x, y):
    a, b, w, h = box
    return a <= x <= a + w and b <= y <= b + h


def _coverage(mask, box):
    h, w = mask.shape
    x0, x1 = max(0, math.ceil(box[0] - .5)), min(w, math.ceil(box[2] - .5))
    y0, y1 = max(0, math.ceil(box[1] - .5)), min(h, math.ceil(box[3] - .5))
    return float(mask[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.


def _member(mask, x, y, tol=3):
    h, w = mask.shape
    if 0 <= x < w and 0 <= y < h and mask[int(y), int(x)]:
        return True, True
    x0, x1 = max(0, math.ceil(x - tol - .5)), min(w, math.floor(x + tol - .5) + 1)
    y0, y1 = max(0, math.ceil(y - tol - .5)), min(h, math.floor(y + tol - .5) + 1)
    yy, xx = np.nonzero(mask[y0:y1, x0:x1])
    near = bool(len(xx)) and float(np.hypot(xx + x0 + .5 - x, yy + y0 + .5 - y).min()) <= tol
    return False, near


def questions_for_image(image, annotations):
    """All eligible (owner, competitor, joint) questions in one COCO image."""
    if image.get('license') not in LICENSES:
        return []
    w, h = image['width'], image['height']
    anns = sorted(annotations, key=lambda a: a['id'])
    refs = {a['id']: reference_box(a) for a in anns}
    masks = {a['id']: polygon_mask(a.get('segmentation'), w, h) for a in anns}
    people = [a for a in anns if refs[a['id']] is not None and masks[a['id']] is not None]
    visible = [(a['id'], *a['keypoints'][3 * j:3 * j + 2]) for a in anns for j in JOINTS
               if len(a.get('keypoints', [])) == 51 and a['keypoints'][3 * j + 2] == 2]
    out = []
    for owner in people:
        oid, ref = owner['id'], refs[owner['id']]
        for j, name in JOINTS.items():
            x, y, v = owner['keypoints'][3 * j:3 * j + 3]
            if v != 2 or not (0 <= x < w and 0 <= y < h) or not _in_xywh(owner['bbox'], x, y):
                continue
            r = .025 * max(w, h)
            target = [x - r, y - r, x + r, y + r]
            if target[0] < 0 or target[1] < 0 or target[2] > w or target[3] > h or _intersects(target, ref):
                continue
            if any(aid != oid and math.hypot(X - x, Y - y) <= 6 for aid, X, Y in visible):
                continue
            exact, near = _member(masks[oid], x, y)
            if not near:
                continue
            for other in people:
                cid, cref = other['id'], refs[other['id']]
                if cid == oid or not _in_xywh(other['bbox'], x, y) or _intersects(target, cref) or _intersects(ref, cref):
                    continue
                if min(_coverage(masks[oid], ref), _coverage(masks[cid], cref)) < .85:
                    continue
                if max(_coverage(masks[cid], ref), _coverage(masks[oid], cref)) > .1:
                    continue
                if not exact and any(masks[a['id']] is not None and _member(masks[a['id']], x, y)[0]
                                     for a in anns if a['id'] != oid):
                    continue
                out.append({'image_id': image['id'], 'joint': name, 'target_xy': [x, y],
                            'owner': {'annotation_id': oid, 'torso_box': ref},
                            'competitor': {'annotation_id': cid, 'torso_box': cref}})
    return out


def nearest_torso(target_xy, boxes):
    """boxes: {'A': xyxy, 'B': xyxy} in pixels. Squared distance to each box (0 inside); exact tie -> unknown."""
    x, y = target_xy
    d = {k: max(a - x, 0, x - c) ** 2 + max(b - y, 0, y - e) ** 2 for k, (a, b, c, e) in boxes.items()}
    return 'unknown' if d['A'] == d['B'] else min(d, key=d.get)


def nearest_tool_point(target_xy, tool_points):
    """tool_points: {'A': [(x, y) or None, ...], 'B': [...]} same-family predictions in pixels."""
    if any(p is None for pts in tool_points.values() for p in pts):
        return 'unknown'
    x, y = target_xy
    d = {k: min((px - x) ** 2 + (py - y) ** 2 for px, py in pts) for k, pts in tool_points.items()}
    return 'unknown' if d['A'] == d['B'] else min(d, key=d.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coco_keypoints_json')
    ap.add_argument('out_jsonl')
    a = ap.parse_args()
    data = json.load(open(a.coco_keypoints_json))
    by_image = {}
    for ann in data['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann)
    n = 0
    with open(a.out_jsonl, 'w') as f:
        for image in data['images']:
            for q in questions_for_image(image, by_image.get(image['id'], [])):
                q['license'] = image['license']
                q['flickr_url'] = image.get('flickr_url')
                f.write(json.dumps(q) + '\n')
                n += 1
    print(n, 'questions')


if __name__ == '__main__':
    main()
