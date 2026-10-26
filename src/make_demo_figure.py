"""Render docs/assets/demo_ownership.jpg: three COCO photos with the ownership question and the answers.

The question, boxes, target point and answers are copied from the saved ownership study (all six option orders,
16 COCO train2017 photos; archive files `ownership_repair/data-v1/queries.json` for the geometry and
`ownership_order/evaluation-v1.json` -> `six_order.per_query` for the answers). No model is run here.

The three photos are CC BY 2.0 on Flickr (checked on the Flickr pages on 2026-10-26). Credits are drawn into the
figure and repeated under it in the README.

Environment: Python 3.10+ with matplotlib and Pillow.

  python3 src/make_demo_figure.py --coco-dir <folder with COCO train2017 jpgs> [--download]

`--download` fetches the three original files from images.cocodataset.org into --coco-dir if they are missing.
"""
import argparse
import urllib.request
from pathlib import Path

from PIL import Image

from make_charts import BLUE, INK, INK2, ORANGE, SURF, plt_setup

# Normalised xyxy torso references, normalised target point, and the saved answers.
QUESTIONS = [
    {'coco_id': 462530, 'part': 'ankle', 'A': [0.2784, 0.231, 0.3403, 0.3746], 'B': [0.3744, 0.5883, 0.4569, 0.7685],
     'target': [0.2766, 0.7465], 'reference': 'A', 'geometry': 'no answer', 'laya': 'B',
     'renamed_raw': 'B', 'credit': '"5V7A9362" by Glenn Brunette', 'flickr': 9431997608},
    {'coco_id': 246285, 'part': 'wrist', 'A': [0.3069, 0.5587, 0.3744, 0.5996], 'B': [0.6359, 0.4571, 0.7297, 0.5429],
     'target': [0.3453, 0.7253], 'reference': 'A', 'geometry': 'A', 'laya': 'B', 'renamed_raw': 'B',
     'credit': '"Zach MacPhee Dives Back into First" by Barry Stahl', 'flickr': 5505334912},
    {'coco_id': 203965, 'part': 'ankle', 'A': [0.4575, 0.3779, 0.5175, 0.5242], 'B': [0.1881, 0.2204, 0.2275, 0.2879],
     'target': [0.3031, 0.8583], 'reference': 'A', 'geometry': 'A', 'laya': 'B', 'renamed_raw': 'B',
     'credit': '"Brett Pill" by Courtney Silvia', 'flickr': 7658431080},
]


def coco_path(folder, coco_id, download):
    path = Path(folder) / f'{coco_id:012d}.jpg'
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'{path} missing (pass --download to fetch it from images.cocodataset.org)')
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f'http://images.cocodataset.org/train2017/{coco_id:012d}.jpg', path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco-dir', required=True)
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'docs/assets/demo_ownership.jpg'))
    a = ap.parse_args()
    plt = plt_setup()
    from matplotlib.patches import Circle, Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6), gridspec_kw={'wspace': 0.05})
    for ax, q in zip(axes, QUESTIONS):
        im = Image.open(coco_path(a.coco_dir, q['coco_id'], a.download)).convert('RGB')
        w, h = im.size
        ax.imshow(im)
        ax.set_axis_off()
        ax.grid(False)
        for name, color in (('A', BLUE), ('B', ORANGE)):
            x0, y0, x1, y1 = q[name]
            ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, ec=color, lw=2.6))
            ax.text(x0 * w, y0 * h - 3, name, color='white', fontsize=11, weight='bold', va='bottom',
                    bbox=dict(boxstyle='square,pad=0.15', fc=color, ec='none'))
        tx, ty = q['target'][0] * w, q['target'][1] * h
        r = 0.028 * max(w, h)
        ax.add_patch(Circle((tx, ty), r, fill=False, ec='white', lw=3.2))
        ax.add_patch(Circle((tx, ty), r, fill=False, ec='#d62f2f', lw=1.6))
        q['lines'] = [(f'Whose {q["part"]} is circled?', INK, 'bold'),
                      (f'COCO reference: {q["reference"]}    pose geometry: {q["geometry"]}', INK2, 'normal'),
                      (f'Laya: {q["laya"]} (wrong)    names A/B swapped: {q["renamed_raw"]} again', INK2, 'normal')]
        ax.text(1.0, 0.01, q['credit'] + ', CC BY 2.0', transform=ax.transAxes, fontsize=6.5, color='white',
                ha='right', va='bottom', bbox=dict(boxstyle='square,pad=0.2', fc='#00000088', ec='none'))
    fig.suptitle('Whose limb is it? The released Laya Vision head answers "B" on 15 of 16 photos', fontsize=13,
                 color=INK, y=0.97)
    fig.text(0.5, 0.035, 'Torso boxes and the circled point are given to the model as coordinates and drawn here only '
             'for the reader. Answers: released head with pose text, all six option orders averaged.\nCOCO train2017 '
             'development photos, not a held-out test. Photos from Flickr via COCO, CC BY 2.0; annotations added.',
             ha='center', fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.25)
    for ax, q in zip(axes, QUESTIONS):
        x0 = ax.get_position().x0
        for k, (text, color, weight) in enumerate(q['lines']):
            fig.text(x0, 0.215 - 0.04 * k, text, fontsize=9.5, color=color, weight=weight, va='top')
    fig.savefig(a.out, dpi=110, facecolor=SURF, pil_kwargs={'quality': 85, 'optimize': True})
    print('wrote', a.out)


if __name__ == '__main__':
    main()
