"""Frozen vision-tower features for the matched encoder comparison (E08).

Every encoder gets the same 448x448 RGB crop (normalised actor box expanded by 10% per side, padded to a square
with black, bicubic; see letterbox_crop). We keep the final patch
tokens of the vision tower only: no connector, no language model, no label text. Two outputs per crop:
  - a mean-pooled vector (posture probe, src/posture_probe.py);
  - a 28x28 token grid (spatial decoder, src/spatial_decoder.py). DINOv2's 32x32 grid is bilinearly resampled to
    28x28; Qwen's grouped 2x2 patch order is restored to row-major.

Encoders (local snapshots under $FSV_MODELS, see src/fsv_paths.py):
  dino  facebook/dinov2-small (CLS token dropped)
  laya  vision tower of thaitea/laya-vision-smolvlm-256m (identical to SmolVLM-256M-Instruct's, see E04)
  qwen  vision tower of Qwen3.8-27B, loaded on its own from a vision-only safetensors file

Environment used in the study: torch 2.12, transformers 5.17, one 6 GB laptop GPU, BF16, batch 1.

  python3 src/encoder_features.py dino crops/*.png --out outputs/features-dino
  python3 src/encoder_features.py laya crops/*.png --model-dir $FSV_MODELS/laya-vision-smolvlm-256m --out ...
  python3 src/encoder_features.py qwen crops/*.png --model-dir <vision weights> --config-dir <Qwen config> --out ...
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 448


def crop_geometry(box, size):
    """Pixel crop for a normalised xyxy actor box expanded by 10% per side, plus the 10 box-geometry features
    used as the 'geometry only' baseline (log aspect, width, height, area, centre, four edge-contact flags)."""
    import math
    x0, y0, x1, y1 = box
    w, h = size
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('invalid normalised box')
    bw, bh = x1 - x0, y1 - y0
    crop = (math.floor(max(0, x0 - .1 * bw) * w + 1e-9), math.floor(max(0, y0 - .1 * bh) * h + 1e-9),
            math.ceil(min(1, x1 + .1 * bw) * w - 1e-9), math.ceil(min(1, y1 + .1 * bh) * h - 1e-9))
    geometry = [math.log(bw * w / (bh * h)), bw, bh, bw * bh, (x0 + x1) / 2, (y0 + y1) / 2,
                int(x0 == 0), int(y0 == 0), int(x1 == 1), int(y1 == 1)]
    return crop, geometry


def letterbox_crop(image, box, size=SIZE):
    """The study's crop: expanded actor box, aspect kept, padded with black to size x size (bicubic)."""
    from PIL import ImageOps
    crop, _ = crop_geometry(box, image.size)
    return ImageOps.pad(image.convert('RGB').crop(crop), (size, size), method=Image.Resampling.BICUBIC,
                        color=(0, 0, 0), centering=(.5, .5))


def row_major_qwen(tokens):
    """Qwen emits 784 tokens in 2x2 merge groups; restore a row-major 28x28 grid."""
    import torch
    t = torch.as_tensor(tokens)
    assert t.shape[0] == 784
    return t.reshape(14, 14, 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(28, 28, -1)


class Encoder:
    def __init__(self, name, model_dir=None, config_dir=None, device='cuda'):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.name, self.device, self.dtype = name, device, torch.bfloat16
        if name == 'dino':
            src = model_dir or 'facebook/dinov2-small'
            self.model = AutoModel.from_pretrained(src, dtype=self.dtype).to(device).eval()
            self.processor = AutoImageProcessor.from_pretrained(src)
        elif name == 'laya':
            from safetensors import safe_open
            from transformers import AutoConfig
            from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer
            config = AutoConfig.from_pretrained(Path(model_dir) / 'backbone').vision_config
            with torch.device('meta'):
                model = Idefics3VisionTransformer(config)
            with safe_open(Path(model_dir) / 'model.safetensors', framework='pt', device='cpu') as f:
                state = {k.removeprefix('encoder.vision_model.'): f.get_tensor(k).to(self.dtype) for k in f.keys()
                         if k.startswith('encoder.vision_model.')}
            model.load_state_dict(state, strict=True, assign=True)
            self.model = model.to(device).eval()
        elif name == 'qwen':
            from safetensors import safe_open
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding
            config = Qwen3_5VisionConfig(**json.loads((Path(config_dir) / 'config.json').read_text())['vision_config'])
            with torch.device('meta'):
                model = Qwen3_5VisionModel(config)
            with safe_open(Path(model_dir) / 'model.safetensors', framework='pt', device='cpu') as f:
                state = {k: f.get_tensor(k) for k in f.keys()}
            model.load_state_dict(state, strict=True, assign=True)
            model.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config)  # non-persistent buffers
            self.model = model.to(device, dtype=self.dtype).eval()
            self.processor = AutoImageProcessor.from_pretrained(config_dir)
        else:
            raise ValueError(name)

    def encode(self, image):
        """Return (mean-pooled vector, 28x28xC grid) as float32 numpy arrays."""
        import torch
        import torch.nn.functional as F
        assert image.size == (SIZE, SIZE)
        with torch.inference_mode():
            if self.name == 'dino':
                px = self.processor(images=image, do_resize=False, do_center_crop=False,
                                    return_tensors='pt')['pixel_values'].to(self.device, self.dtype)
                tokens = self.model(pixel_values=px, interpolate_pos_encoding=True).last_hidden_state[0, 1:]
                grid = tokens.reshape(32, 32, -1).permute(2, 0, 1)[None].float()
                grid = F.interpolate(grid, size=(28, 28), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
            elif self.name == 'laya':
                px = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)[None].float().div(255).sub(.5).div(.5)
                tokens = self.model(pixel_values=px.to(self.device, self.dtype)).last_hidden_state[0]
                side = int(round(tokens.shape[0] ** 0.5))
                grid = tokens.reshape(side, side, -1).float()
            else:
                inputs = self.processor(images=image, do_resize=False, return_tensors='pt').to(self.device)
                assert inputs['image_grid_thw'].tolist() == [[1, 28, 28]]
                tokens = self.model(hidden_states=inputs['pixel_values'].to(self.dtype),
                                    grid_thw=inputs['image_grid_thw']).last_hidden_state
                grid = row_major_qwen(tokens.float())
            vector = tokens.float().mean(0)
        return vector.cpu().numpy(), grid.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('encoder', choices=['dino', 'laya', 'qwen'])
    ap.add_argument('crops', nargs='+', help='448x448 crops (see letterbox_crop)')
    ap.add_argument('--model-dir')
    ap.add_argument('--config-dir')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    enc = Encoder(a.encoder, a.model_dir, a.config_dir, a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vectors, grids, seconds = [], [], []
    for path in a.crops:
        image = Image.open(path).convert('RGB')
        t = time.monotonic()
        v, g = enc.encode(image)
        seconds.append(time.monotonic() - t)
        vectors.append(v)
        grids.append(g.astype(np.float16))
    np.savez(out / 'features.npz', vectors=np.stack(vectors), grids=np.stack(grids))
    (out / 'index.json').write_text(json.dumps({'encoder': a.encoder, 'crops': a.crops,
                                                'median_seconds': float(np.median(seconds))}, indent=2))
    print(f'{len(a.crops)} crops, median {1000 * np.median(seconds):.1f} ms')


if __name__ == '__main__':
    main()
