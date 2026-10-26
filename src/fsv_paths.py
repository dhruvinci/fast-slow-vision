"""Where data, model weights and outputs live. Set these environment variables; nothing is hard-coded.

  FSV_DATA     datasets you have downloaded yourself (COCO, Penn Action, ...)    default ./data
  FSV_MODELS   local model snapshots (Laya Vision, SmolVLM, DINOv2, Qwen tower)    default ./models
  FSV_OUT      experiment outputs (features, fits, reports)                         default ./outputs

None of the data or weights are redistributed in this repository; see the README for sources and licences.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _root(name, default):
    return Path(os.environ.get(name, REPO / default)).expanduser().resolve()


DATA = _root('FSV_DATA', 'data')
MODELS = _root('FSV_MODELS', 'models')
OUT = _root('FSV_OUT', 'outputs')

# Pinned revisions used in the study (see experiments/E04-component-audit).
LAYA_VISION_REPO = 'thaitea/laya-vision-smolvlm-256m'
LAYA_VISION_WEIGHTS_REVISION = 'a2653db2831b9b03b875a6558618feda833b7ff0'
LAYA_VISION_CODE = 'https://github.com/r33drichards/laya-vision'
LAYA_VISION_CODE_REVISION = '86ccec115ef3d72d1851168fc9b7194e8dbed35a'
SMOLVLM_REPO = 'HuggingFaceTB/SmolVLM-256M-Instruct'
SMOLVLM_REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
DINOV2_SMALL_REPO = 'facebook/dinov2-small'
DINOV2_SMALL_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
