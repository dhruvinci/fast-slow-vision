"""Synthetic checks of the published modules (no data or model weights needed).

  python3 -m unittest discover -s tests -v
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class Controls(unittest.TestCase):
    def test_consistency_and_identity_average(self):
        import controls
        p = {'A': .2, 'B': .7, 'unknown': .1}
        q = {'A': .6, 'B': .3, 'unknown': .1}
        self.assertEqual(controls.consistency([(p, p), (p, q)]), (1, 2))
        self.assertEqual(controls.consistency([(p, q), (p, p)], {'A': 'B', 'B': 'A'}), (1, 2))
        runs = [{'A': 1., 'B': 3., 'unknown': 0.}, {'A': 2., 'B': 2., 'unknown': 0.}]
        renamed = [{'A': 3., 'B': 1., 'unknown': 0.}]  # raw logits under swapped names; mapped back before averaging
        avg = controls.identity_average(runs, renamed)
        self.assertAlmostEqual(avg['B'], (3 + 2 + 3) / 3)
        self.assertEqual(len(controls.all_orders(['A', 'B', 'unknown'])), 6)

    def test_directional(self):
        import controls
        inc = {'increased': .8, 'decreased': .1, 'unchanged': .1}
        dec = {'increased': .1, 'decreased': .8, 'unchanged': .1}
        unc = {'increased': .1, 'decreased': .1, 'unchanged': .8}
        self.assertEqual(controls.directional_both_correct([(inc, dec), (unc, unc)], ['increased', 'unchanged']),
                         (1, 1))


class Movement(unittest.TestCase):
    def test_pair_midpoint_and_policy(self):
        import limb_pair_movement as m
        pts0 = [None] * 17
        pts1 = [None] * 17
        for j in (9, 10, 11, 12):
            pts0[j] = (100., 100.)
            pts1[j] = (100., 100.)
        pts1[9], pts1[10] = (100., 160.), (100., 140.)   # wrists move down 50 px on average
        present = [[p is not None for p in pts0], [p is not None for p in pts1]]
        d = m.movement([pts0, pts1], present, 'wrists', 300, 400)
        self.assertAlmostEqual(d[1], 50 / 500)
        self.assertEqual([m.classify(v) for v in d], [1, 2])
        rows = [{'returned': True, 'correct': i < 9, 'confidence': 1 - i / 10} for i in range(10)]
        policy = m.select_policy(rows)
        self.assertTrue(policy['feasible'])
        self.assertLessEqual(policy['error'], .15)


class Models(unittest.TestCase):
    def test_spatial_decoder(self):
        import torch
        import spatial_decoder as s
        model = s.Decoder()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 24738)
        x, q = torch.randn(2, 64, 28, 28), torch.randn(2, 2, 64)
        a, b = model(x, q), model(x, q.flip(1))
        self.assertTrue(torch.equal(a[:, [0, 2, 1]], b))   # swapping queries swaps actors exactly

    def test_temporal_completion_starts_at_interpolation(self):
        import torch
        import temporal_completion as t
        model = t.TemporalCompletion()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 68280)
        rng = np.random.default_rng(0)
        poses = rng.random((9, 12, 2)).astype(np.float32)
        p, v, _ = t.corrupt(poses, np.ones((9, 12), bool), 'frame_gap')
        d = t.prepare_inputs(p, v, np.arange(9.))
        out = model(torch.from_numpy(d['input'])[None], torch.from_numpy(d['baseline'])[None])
        self.assertTrue(torch.allclose(out[0], torch.from_numpy(d['baseline'])))

    def test_posture_probe(self):
        import posture_probe as pp
        rng = np.random.default_rng(1)
        x = rng.normal(size=(120, 80))
        y = rng.integers(0, 4, 120)
        model = pp.fit(x, y, [f'f{i % 6}' for i in range(120)])
        self.assertEqual(model['coef'].size + model['intercept'].size, 260)
        self.assertEqual(pp.predict(model, x[:5]).shape, (5, 4))


class Ownership(unittest.TestCase):
    def test_geometry_baselines(self):
        import ownership_questions as o
        boxes = {'A': [0, 0, 10, 10], 'B': [50, 50, 60, 60]}
        self.assertEqual(o.nearest_torso((12, 12), boxes), 'A')
        self.assertEqual(o.nearest_tool_point((5, 5), {'A': [(6, 6), None], 'B': [(50, 50), (40, 40)]}), 'unknown')
        self.assertEqual(o.nearest_tool_point((5, 5), {'A': [(6, 6), (9, 9)], 'B': [(50, 50), (40, 40)]}), 'A')


class ComponentAudit(unittest.TestCase):
    def _write(self, path, tensors):
        header, blob, offset = {}, b'', 0
        for name, (dtype, arr) in tensors.items():
            data = arr.tobytes()
            header[name] = {'dtype': dtype, 'shape': list(arr.shape), 'data_offsets': [offset, offset + len(data)]}
            blob += data
            offset += len(data)
        h = json.dumps(header).encode()
        Path(path).write_bytes(struct.pack('<Q', len(h)) + h + blob)

    def test_bf16_exact_equality(self):
        import component_audit as c
        x = np.array([1.5, -2.0, 0.25, 3.0], dtype=np.float32)
        bf16 = (x.view('<u4') >> 16).astype('<u2')   # these values are exact in BF16
        with tempfile.TemporaryDirectory() as d:
            self._write(f'{d}/a.safetensors', {'encoder.vision_model.w': ('F32', x)})
            self._write(f'{d}/b.safetensors', {'model.vision_model.w': ('BF16', bf16)})
            r = c.compare_component(c.SafeFile(f'{d}/a.safetensors'), 'encoder.vision_model.',
                                    c.SafeFile(f'{d}/b.safetensors'), 'model.vision_model.')
        self.assertEqual(r['different_elements'], 0)
        self.assertEqual(r['compared'], 1)


if __name__ == '__main__':
    unittest.main()
