# Experiments

One folder per experiment, numbered in the order the work ran (21–28 September 2026). Each experiment grouped
several fixed, pre-declared stages; each README gives the question, setup, result, verdict and what was learned. The
original reports (with frozen protocols, hashes and receipts) live in the research archive; line references for every
number are in [`docs/SOURCES.md`](../docs/SOURCES.md). All labels are AI references or published dataset annotations,
not human ground truth.

| ID | Question | Verdict |
|---|---|---|
| [E00](E00-pilot-typed-heads/) | Does relational supervision improve Laya Vision's typed decision heads on posture? | **fail** (heads 50.0%; frozen-feature probe 66.7%; prior 36.1%) |
| [E01](E01-grounding-controls/) | Do the answers depend on the right person, option, image and frame order? | **fail** (option order 42.4%, actor swap 5.3%, blank image kept 36/41) |
| [E02](E02-ownership-chain/) | Whose wrist or ankle is it? Seven Laya-based readouts against geometry | **fail** (none beats geometry; the one that ties ignores the photo) |
| [E03](E03-spatial-readouts/) | Can small decoders on Laya feature grids find a given person's joints? | **fail** (≤ 23%; pretrained pose tool 98% on COCO) |
| [E04](E04-component-audit/) | Which parts of Laya Vision differ from SmolVLM? | vision tower identical; connector and language model changed |
| [E05](E05-limb-pair-movement/) | Hip-relative movement of wrists and ankles | named joints **fail**; limb-pair midpoints **pass** (error −62% vs no movement) |
| [E06](E06-temporal-completion/) | Does a learned temporal adapter fill hidden poses better than interpolation? | **fail** (interpolation selected in both runs) |
| [E07](E07-fight-pilots/) | Actor posture on new fighting sources; does temporal context help? | pilot 1 **fail**; DINOv2 readout **passes** a fresh audit (0.79); learned temporal **fail** |
| [E08](E08-encoder-comparison/) | DINOv2 vs Laya (SmolVLM) vs Qwen3.8 towers under identical inputs | no universal winner; Laya's tower trails on posture |

Not run (prepared or partial):
- an ownership training-size curve (32 / 96 / 288 photos): data prepared, no run;
- the full fighting-label batch: 5 of 34 planned requests saved, one invalid; no full temporal model;
- blind human review of the posture, ownership and contact references.
