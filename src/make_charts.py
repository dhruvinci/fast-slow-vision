"""Render the result charts in docs/assets/ from the numbers reported in the experiment write-ups.

Every number below is copied from a saved result report in the research archive; the comment above each chart
names the file and line (paths relative to the archive's `Laya Vision/` folder, listed again in docs/SOURCES.md).
Nothing is recomputed here.

Environment: any Python 3.10+ with matplotlib >= 3.8 and numpy.

  python3 src/make_charts.py [--out docs/assets]
"""
import argparse
from pathlib import Path

import numpy as np

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GREY, INK, INK2, SURF = '#9a9993', '#0b0b0b', '#52514e', '#fcfcfb'


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GREY, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                         'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': INK, 'legend.frameon': False,
                         'axes.axisbelow': True})
    return plt


def label_bars(ax, bars, fmt='{:.1f}', horizontal=False, pad=0.8, size=8):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v + pad, b.get_y() + b.get_height() / 2, fmt.format(v), va='center', ha='left', fontsize=size,
                    color=INK2)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt.format(v), va='bottom', ha='center', fontsize=size,
                    color=INK2)


def charts(plt, out):
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)

    # E00 pilot: posture agreement with AI-consensus references, 36 held-out questions.
    # research/RESULTS.md:33-40; RESEARCH_SUMMARY.md:42-48
    rows = [('training class prior (no image)', 36.1, GREY), ('released Laya Vision', 36.1, GREY),
            ('contextual marker probe', 38.9, AQUA), ('coarse posture head', 50.0, BLUE),
            ('posture + relations head', 50.0, BLUE), ('frozen visual probe', 66.7, AQUA)]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(rows))
    bars = ax.barh(y, [r[1] for r in rows], height=0.62, color=[r[2] for r in rows])
    label_bars(ax, bars, '{:.1f}%', horizontal=True)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel('agreement with AI-consensus labels (%), 36 test questions')
    ax.set_title('Pilot: Laya decision heads vs a frozen-feature probe', fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label='baselines'), Patch(color=BLUE, label='trained Laya decision heads'),
                       Patch(color=AQUA, label='probes on frozen features')], fontsize=8, loc='lower right')
    save(fig, 'chart_pilot.png')

    # E01 controls on the posture + relations head. research/RESULTS.md:70-78; RESEARCH_SUMMARY.md:52
    # blank image: 36 of 41 previously correct answers unchanged -> 5/41 changed.
    rows = [('options reordered:\nsame answer', 39 / 92 * 100, 90, '39/92'),
            ('people A/B swapped:\nanswer follows the person', 1 / 19 * 100, 80, '1/19'),
            ('image blanked:\ncorrect answer changes', 5 / 41 * 100, 100, '5/41'),
            ('clip reversed: direction\nright both ways', 0.0, 100, '0/7')]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.55, color=BLUE)
    for yy, (_, v, t, frac) in zip(y, rows):
        ax.text(v + 1.2, yy, f'{v:.1f}%  ({frac})', va='center', fontsize=8, color=INK2)
        ax.plot([t, t], [yy - 0.36, yy + 0.36], color=INK2, lw=2)
    ax.text(90, y[0] + 0.45, 'target', ha='center', va='bottom', fontsize=7.5, color=INK2)
    ax.text(80, y[1] + 0.45, 'target', ha='center', va='bottom', fontsize=7.5, color=INK2)
    ax.text(99, y[2] + 0.45, 'ideal', ha='right', va='bottom', fontsize=7.5, color=INK2)
    ax.text(99, y[3] + 0.45, 'ideal', ha='right', va='bottom', fontsize=7.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.set_xlabel('share of test questions that behave as a grounded model must (%)')
    ax.set_title('Grounding controls on the posture + relations head', fontsize=11)
    save(fig, 'chart_controls.png')

    # E02 ownership chain. Geometry = nearest torso; readout = Laya-based; blank = same readout, photo removed.
    # ownership_reference/RESULTS.md:17,23,52 (172 questions; heads unchanged under blank images)
    # ownership_order/RESULTS.md:13-17 (16 photos, all six option orders)
    # ownership_markers/RESULTS.md:9-13 (drawn markers)
    # ownership_pairs/RESULTS.md:7-11 (32 development photos, both targets correct)
    # ownership_locality/RESULTS.md:24,30-31
    stages = ['frozen features,\nfitted heads\n(172 questions)', 'released head,\nall six orders\n(16 photos)',
              '+ drawn boxes\nand marker\n(16 photos)', 'fine-tuned head\n(32 new photos,\nboth targets)',
              'compact local\nscorer (32 new\nphotos, both)']
    geometry = [125 / 172, 13 / 16, 13 / 16, 24 / 32, 24 / 32]
    model = [90 / 172, 6 / 16, 5 / 16, 6 / 32, 24 / 32]
    blank = [90 / 172, 7 / 16, 7 / 16, 5 / 32, 24 / 32]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    x = np.arange(len(stages))
    for k, (vals, c, lab) in enumerate(((geometry, GREY, 'geometry baseline (no Laya features)'),
                                        (model, BLUE, 'Laya-based readout'),
                                        (blank, ORANGE, 'same readout, photo removed'))):
        bars = ax.bar(x + (k - 1) * 0.26, np.array(vals) * 100, 0.24, color=c, label=lab)
        label_bars(ax, bars, '{:.0f}', pad=1.0, size=7.5)
    ax.set_xticks(x, stages, fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% correct (COCO reference)')
    ax.set_title('Whose wrist or ankle is it? On photos it was not trained on, no Laya readout beats geometry', fontsize=10.5)
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.27))
    save(fig, 'chart_ownership_chain.png')

    # E03 joint localisation on the same 16 COCO photos. pose_reference_pilot/RESULTS.md:13-16,24-27
    methods = ['geometry head', 'local-feature head', 'person-reference\nhead', 'YOLOX-M +\nRTMPose-M']
    pck = [13 / 116, 18 / 116, 23 / 116, 114 / 116]
    both = [4 / 44, 1 / 44, 1 / 44, 42 / 44]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(len(methods))
    b1 = ax.bar(x - 0.2, np.array(pck) * 100, 0.38, color=BLUE, label='wrist/ankle within 5% of image diagonal')
    b2 = ax.bar(x + 0.2, np.array(both) * 100, 0.38, color=ORANGE, label="both people's joint placed correctly")
    label_bars(ax, b1, '{:.0f}', pad=1.0)
    label_bars(ax, b2, '{:.0f}', pad=1.0)
    ax.set_xticks(x, methods, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('% of visible targets')
    ax.text(1.0, -0.33, 'first three: small decoders on Laya feature grids', transform=ax.transAxes, fontsize=7.5,
            color=INK2, ha='right')
    ax.set_title('Finding joints: decoders on Laya grids vs an off-the-shelf pose tool', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    save(fig, 'chart_spatial_readouts.png')

    # E05 movement error vs predicting no movement (same support), % of image diagonal.
    # temporal_reference/RESULTS.md:43,46; pose_transfer/RESULTS.md:27-28;
    # centroid_diagnostics/RESULTS.md:21,23; centroid_transfer/RESULTS.md:39
    groups = ['decoder on Laya\ngrids, named joints\n(Penn test)', 'pose tool,\nnamed joints\n(Penn confirmation)',
              'pose tool,\nlimb-pair midpoints\n(development)', 'pose tool,\nlimb-pair midpoints\n(confirmation)']
    method = [3.758, 2.180, 0.564, 0.619]
    zero = [2.127, 2.095, 1.011, 1.631]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    x = np.arange(len(groups))
    b1 = ax.bar(x - 0.2, zero, 0.38, color=GREY, label='predict no movement')
    b2 = ax.bar(x + 0.2, method, 0.38, color=BLUE, label='model')
    label_bars(ax, b1, '{:.2f}', pad=0.04)
    label_bars(ax, b2, '{:.2f}', pad=0.04)
    ax.set_xticks(x, groups, fontsize=8.5)
    ax.set_ylim(0, 4.3)
    ax.set_ylabel('mean axis error\n(% of image diagonal, lower is better)')
    ax.set_title('Hip-relative movement over 8 frames: only limb-pair midpoints beat "no movement"', fontsize=10.5)
    ax.legend(fontsize=8)
    save(fig, 'chart_movement.png')

    # E06 temporal completion, reused Penn test, same returned joints. temporal_completion/RESULTS.md:21-25
    tasks = ['one middle frame\nhidden', 'three middle frames\nhidden', 'middle wrists and\nankles hidden']
    hold = [0.461, 0.841, 0.553]
    lin = [0.187, 0.338, 0.205]
    learned = [0.238, 0.386, 0.263]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(len(tasks))
    for k, (vals, c, lab) in enumerate(((hold, GREY, 'hold last pose'), (lin, BLUE, 'linear interpolation'),
                                        (learned, ORANGE, 'trained adapter (rejected)'))):
        bars = ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=c, label=lab)
        label_bars(ax, bars, '{:.2f}', pad=0.01, size=7.5)
    ax.set_xticks(x, tasks, fontsize=8.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel('mean joint error\n(% of image diagonal)')
    ax.set_title('Filling hidden poses: plain interpolation beats the learned adapter', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    save(fig, 'chart_temporal.png')

    # E07 fight pilots. Left: fresh audit, 47 known AI-reference slots. fight_experiment/pilot2-audit/RESULTS.md:3,11,13
    # Right: hidden-centre probe, 84 family-holdout targets. fight_experiment/pilot2-temporal/RESULTS.md:7-11
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    rows = [('DINOv2 readout', 0.7892, BLUE), ('Laya-feature readout', 0.2407, ORANGE),
            ('geometry only', 0.3599, GREY), ('training majority', 0.25, GREY), ('blank image', 0.25, GREY),
            ('shuffled labels', 0.2353, GREY)]
    y = np.arange(len(rows))[::-1]
    bars = a1.barh(y, [r[1] for r in rows], height=0.6, color=[r[2] for r in rows])
    label_bars(a1, bars, '{:.2f}', horizontal=True, pad=0.01)
    a1.set_yticks(y, [r[0] for r in rows])
    a1.set_xlim(0, 1)
    a1.set_xlabel('macro recall vs AI references')
    a1.set_title('Fresh 3-family audit (frozen readout)', fontsize=10.5)
    rows = [('interpolate neighbours', 0.8247, GREY), ('carry previous state', 0.7579, GREY),
            ('learned, offline', 0.6151, BLUE), ('learned, causal', 0.4835, BLUE), ('no temporal context', 0.2130, GREY)]
    y = np.arange(len(rows))[::-1]
    bars = a2.barh(y, [r[1] for r in rows], height=0.6, color=[r[2] for r in rows])
    label_bars(a2, bars, '{:.2f}', horizontal=True, pad=0.01)
    a2.set_yticks(y, [r[0] for r in rows])
    a2.set_xlim(0, 1)
    a2.set_xlabel('macro recall on hidden frames')
    a2.set_title('Hidden middle frame: learned heads lose', fontsize=10.5)
    save(fig, 'chart_fight_pilots.png')

    # E08 posture, identical 448x448 crops, 172 known references. encoder_comparison/RESULTS.md:11-13
    enc = ['DINOv2-small', 'Laya (SmolVLM) tower', 'Qwen3.8 tower']
    vals = {'accuracy': [82.0, 70.9, 75.6], 'pooled macro recall': [69.0, 54.0, 74.2],
            'equal-family macro recall': [74.6, 65.7, 66.5]}
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(3)
    for k, (name, c) in enumerate(zip(vals, (BLUE, ORANGE, AQUA))):
        bars = ax.bar(x + (k - 1) * 0.26, vals[name], 0.24, color=c, label=name)
        label_bars(ax, bars, '{:.0f}', pad=0.8, size=7.5)
    ax.set_xticks(x, [f'{e}\n{t} ms/crop' for e, t in zip(enc, (14.4, 23.4, 79.4))], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% (AI references)')
    ax.set_title('Posture from frozen towers, same crops and same 260-parameter head', fontsize=10.5)
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    save(fig, 'chart_encoders_posture.png')

    # E08 spatial ownership, 4 grappling clips, source held out. spatial_encoder_comparison/RESULTS.md:13-15
    rep = ['DINOv2-small', 'Qwen3.8 tower', 'DINO + Qwen']
    vals = {'actor overlap (sparse IoU)': [45.6, 45.5, 57.1], 'macro recall': [73.6, 73.7, 80.9],
            'contact ownership': [60.5, 50.2, 31.3]}
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for k, (name, c) in enumerate(zip(vals, (BLUE, ORANGE, AQUA))):
        bars = ax.bar(x + (k - 1) * 0.26, vals[name], 0.24, color=c, label=name)
        label_bars(ax, bars, '{:.0f}', pad=0.8, size=7.5)
    ax.set_xticks(x, rep, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% (sparse AI references)')
    ax.set_title('Who owns each pixel: fusion helps body cores, hurts contact', fontsize=10.5)
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    save(fig, 'chart_encoders_spatial.png')


def component_diagram(plt, out):
    """E04: which parts of the released Laya Vision checkpoint differ from public SmolVLM-256M-Instruct.

    Vision tower and connector: temporal_reference/parent_identity/IDENTITY_AUDIT.md:18-21.
    Language model: re-run with src/component_audit.py on 2026-10-26 while preparing this repository.
    Decision head size: ownership_supervised/RESULTS.md:9.
    """
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 30)
    boxes = [(1, 'image', 'pixels', '#ffffff', INK2, 8),
             (12, 'vision tower', '197 tensors\n86,433,024 values\nall identical', '#ecebe6', GREY, 24),
             (39, 'connector', '7,077,888 values\n7,077,887 differ\n(relative L2 6.8%)', '#fbe3d9', ORANGE, 23),
             (65, 'language model', '272 tensors, 134.6M values\nall tensors differ\n(relative L2 0.85%)',
              '#fbe3d9', ORANGE, 27),
             (95, 'typed decision\nhead', 'choice / score / noul\n8.3M parameters\nLaya only', '#dce9f8', BLUE,
              24)]
    for x0, title, sub, face, edge, w in boxes:
        ax.add_patch(FancyBboxPatch((x0, 5), w, 18, boxstyle='round,pad=0.4,rounding_size=1.5', fc=face, ec=edge,
                                    lw=1.6))
        ax.text(x0 + w / 2, 19.0, title, ha='center', va='center', fontsize=10, color=INK, weight='bold')
        ax.text(x0 + w / 2, 10.6, sub, ha='center', va='center', fontsize=8, color=INK2)
    for xa, xb in ((9.4, 11.6), (36.4, 38.6), (62.4, 64.6), (92.4, 94.6)):
        ax.annotate('', xy=(xb, 14), xytext=(xa, 14), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.4))
    ax.text(60, 28.0, 'Released Laya Vision weights vs public SmolVLM-256M-Instruct, tensor by tensor', ha='center',
            fontsize=11, color=INK)
    ax.text(60, 0.6, 'grey: identical to SmolVLM   orange: values changed   blue: added by Laya Vision', ha='center',
            fontsize=8, color=INK2)
    fig.savefig(out / 'diagram_components.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'docs' / 'assets'))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    charts(plt, out)
    component_diagram(plt, out)
    print('wrote charts to', out)


if __name__ == '__main__':
    main()
