"""Compare two safetensors checkpoints component by component, exactly, without building a model (E04).

Used to check which parts of the released Laya Vision checkpoint differ from public SmolVLM-256M-Instruct.
Tensors are streamed in chunks (NumPy only, low memory). BF16 is widened to FP32 by shifting bits, so equality is
exact, with no tolerance.

  python3 src/component_audit.py LAYA/model.safetensors SMOLVLM/model.safetensors \
      --component vision_model:encoder.vision_model.:model.vision_model. \
      --component connector:encoder.connector.:model.connector. --out outputs/component_audit.json

Each --component is name:prefix_in_first:prefix_in_second. The two prefixes above are the ones used in the study.
Environment: Python 3.10+, numpy.
"""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

DTYPES = {'F32': ('<f4', 4), 'F16': ('<f2', 2), 'BF16': ('<u2', 2), 'F64': ('<f8', 8)}


class SafeFile:
    """Minimal safetensors reader: validates the header and reads tensor slices lazily."""

    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with self.path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('missing safetensors header')
            length = struct.unpack('<Q', raw)[0]
            if length > 16 * 1024 ** 2 or length + 8 > size:
                raise ValueError('invalid header size')
            header = json.loads(f.read(length))
        self.offset = 8 + length
        self.metadata = header.pop('__metadata__', {})
        self.header = header
        for key, item in header.items():
            if item['dtype'] not in DTYPES:
                raise ValueError(f'unsupported dtype {item["dtype"]} in {key}')
            start, end = item['data_offsets']
            if math.prod(item['shape']) * DTYPES[item['dtype']][1] != end - start or self.offset + end > size:
                raise ValueError(f'payload size mismatch in {key}')

    def read(self, handle, key, start, count):
        item = self.header[key]
        dtype, width = DTYPES[item['dtype']]
        handle.seek(self.offset + item['data_offsets'][0] + start * width)
        raw = handle.read(count * width)
        if len(raw) != count * width:
            raise ValueError('truncated tensor payload')
        values = np.frombuffer(raw, dtype=dtype)
        if item['dtype'] == 'BF16':  # exact widening: BF16 is the top 16 bits of an FP32
            return (values.astype('<u4') << 16).view('<f4')
        return values.astype(np.float32)


def compare_tensor(left, left_key, right, right_key, chunk=65536):
    a, b = left.header[left_key], right.header[right_key]
    if a['shape'] != b['shape']:
        raise ValueError('tensor shape mismatch')
    n = math.prod(a['shape'])
    different = maximum = square = aa = bb = dot = 0.
    with left.path.open('rb') as lf, right.path.open('rb') as rf:
        for start in range(0, n, chunk):
            count = min(chunk, n - start)
            x = left.read(lf, left_key, start, count)
            y = right.read(rf, right_key, start, count)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError('nonfinite weights')
            different += int(np.count_nonzero(x != y))
            xd, yd = x.astype(np.float64), y.astype(np.float64)
            d = xd - yd
            maximum = max(maximum, float(np.abs(d).max()) if count else 0.)
            square += float(d @ d)
            aa += float(xd @ xd)
            bb += float(yd @ yd)
            dot += float(xd @ yd)
    return {'shape': a['shape'], 'dtypes': [a['dtype'], b['dtype']], 'elements': n, 'different_elements': int(different),
            'max_abs_error': maximum, 'sum_squared_error': square, 'l2_first': math.sqrt(aa), 'l2_second': math.sqrt(bb),
            'cosine': dot / math.sqrt(aa * bb) if aa and bb else None}


def compare_component(left, left_prefix, right, right_prefix):
    la = {k[len(left_prefix):]: k for k in left.header if k.startswith(left_prefix)}
    rb = {k[len(right_prefix):]: k for k in right.header if k.startswith(right_prefix)}
    shared = sorted(set(la) & set(rb))
    tensors = {k: compare_tensor(left, la[k], right, rb[k]) for k in shared}
    n = sum(t['elements'] for t in tensors.values())
    sse = sum(t['sum_squared_error'] for t in tensors.values())
    aa = sum(t['l2_first'] ** 2 for t in tensors.values())
    return {'tensors_first': len(la), 'tensors_second': len(rb), 'compared': len(tensors),
            'missing_in_second': sorted(set(la) - set(rb)), 'extra_in_second': sorted(set(rb) - set(la)),
            'elements': n, 'different_elements': sum(t['different_elements'] for t in tensors.values()),
            'max_abs_error': max((t['max_abs_error'] for t in tensors.values()), default=None),
            'rms_error': math.sqrt(sse / n) if n else None,
            'relative_l2_to_first': math.sqrt(sse / aa) if aa else None, 'per_tensor': tensors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('first')
    ap.add_argument('second')
    ap.add_argument('--component', action='append', required=True, help='name:prefix_first:prefix_second')
    ap.add_argument('--out')
    a = ap.parse_args()
    left, right = SafeFile(a.first), SafeFile(a.second)
    report = {}
    for spec in a.component:
        name, lp, rp = spec.split(':')
        report[name] = compare_component(left, lp, right, rp)
    summary = {k: {n: v for n, v in r.items() if n != 'per_tensor'} for k, r in report.items()}
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
