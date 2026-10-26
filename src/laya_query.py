"""Ask the released Laya Vision model a typed ownership question, with the study's controls (E01, E02).

Laya Vision is third-party work (r33drichards/laya-vision, Apache-2.0 code; weights on the Hugging Face Hub as
thaitea/laya-vision-smolvlm-256m, marked non-commercial). It is not part of this repository. Install the pinned
code and download the pinned weights yourself:

  pip install "git+https://github.com/r33drichards/laya-vision@86ccec115ef3d72d1851168fc9b7194e8dbed35a"
  huggingface-cli download thaitea/laya-vision-smolvlm-256m \
      --revision a2653db2831b9b03b875a6558618feda833b7ff0 --local-dir $FSV_MODELS/laya-vision-smolvlm-256m

The question wording and the state layout follow the study. Each question is scored over all six option orders
(the model's own `n_permutations`, which averages logits in label order). Controls:
  - renamed: A and B swap names; the answer is mapped back. A grounded model gives the same person.
  - blank:   the photo is replaced by a grey image; the coordinates stay. A grounded model should lose confidence.

  python3 src/laya_query.py photo.jpg --a 0.31 0.56 0.37 0.60 --b 0.64 0.46 0.73 0.54 --target 0.345 0.725 \
      --part wrist [--model $FSV_MODELS/laya-vision-smolvlm-256m]

Coordinates are fractions of image width/height (x0 y0 x1 y1 for torso boxes, x y for the target point).
On the middle photo of docs/assets/demo_ownership.jpg (COCO train2017 image 246285, the example above) this
prints B for the original, B again after renaming (so 'A' once mapped back) and B with a grey image (run on CPU,
2026-10-26). The study's prompts also carried pose-tool points, so its saved probabilities differ slightly.
Environment: the laya package above (torch, transformers), Pillow.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from fsv_paths import MODELS

OPTIONS = {'A': 'Person A', 'B': 'Person B', 'unknown': 'Cannot determine'}


def ownership_question(part):
    return {'type': 'choice',
            'instructions': f'Which person owns the visible {part} at target.xy? A and B are located by their torso '
                            'boxes. Follow the target body part in the image to its owner. Choose unknown if '
                            'ownership is unclear.',
            'criteria': dict(OPTIONS)}


def ownership_state(image, people, target, part, renamed=False, blank=False):
    if renamed:
        people = {'A': people['B'], 'B': people['A']}
    if blank:
        image = Image.new('RGB', image.size, (128, 128, 128))
    return {'image': image, 'coordinates': 'x/width,y/height in original image', 'people': people,
            'target': {'family': part, 'xy': target}}


def ask(agent, state, question, n_permutations=6):
    answer = agent.predict(state, {'q': question}, n_permutations=n_permutations)['answers']['q']
    return answer['probabilities']


def run(agent, image, people, target, part):
    q = ownership_question(part)
    original = ask(agent, ownership_state(image, people, target, part), q)
    renamed = ask(agent, ownership_state(image, people, target, part, renamed=True), q)
    renamed_mapped = {{'A': 'B', 'B': 'A'}.get(k, k): v for k, v in renamed.items()}
    blank = ask(agent, ownership_state(image, people, target, part, blank=True), q)
    top = lambda p: max(sorted(p), key=p.get)  # noqa: E731
    return {'original': original, 'renamed_mapped_back': renamed_mapped, 'blank_image': blank,
            'answer': top(original), 'answer_after_renaming': top(renamed_mapped), 'answer_blank': top(blank),
            'follows_the_person': top(original) == top(renamed_mapped)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--a', type=float, nargs=4, required=True)
    ap.add_argument('--b', type=float, nargs=4, required=True)
    ap.add_argument('--target', type=float, nargs=2, required=True)
    ap.add_argument('--part', default='wrist', choices=['wrist', 'ankle'])
    ap.add_argument('--model', default=str(Path(MODELS) / 'laya-vision-smolvlm-256m'))
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    import laya
    agent = laya.load_vlm(a.model, device=a.device)
    image = Image.open(a.image).convert('RGB')
    print(json.dumps(run(agent, image, {'A': a.a, 'B': a.b}, a.target, a.part), indent=2))


if __name__ == '__main__':
    main()
