# Findings log

One line per finding, in the order found. Each links to the experiment that supports it. Dates follow the saved
reports (several receipts are in UTC, so late work can fall on the next day in India). All labels are AI references
or published dataset annotations, not human ground truth.

## 2026-09-21 / 22
- Fine-tuning Laya Vision's typed decision head with relational supervision added nothing on posture (+0.0045 macro-F1
  against a pre-registered +0.05); both heads got the same 18 of 36 right. A probe on frozen features scored 66.7%.
  ([E00](experiments/E00-pilot-typed-heads/))
- Confidence thresholds set on calibration footage did not hold on test: 29.4% and 33.3% error instead of 10%.
  ([E00](experiments/E00-pilot-typed-heads/))
- Frozen features carry action information: 80.0% on 120 UT-Interaction clips across held-out recording groups.
  ([E00](experiments/E00-pilot-typed-heads/))
- The trained head kept the same answer under option reordering only 42.4% of the time, followed the person under an
  A/B swap 5.3% of the time, and kept 36 of 41 correct answers with a blank image. Four frames gave the same answers as
  the last frame alone. ([E01](experiments/E01-grounding-controls/))
- Whose wrist or ankle is it? On 172 COCO questions learned geometry got 72.7%, every visual head 52.3% (they never
  left their zero start). A blind AI model with richer views got 40/40 of a 40-question subset.
  ([E02](experiments/E02-ownership-chain/))
- The released full model answers "B" on 15 of 16 photos and 85 of 96 balanced option orders; it follows the letter,
  not the person. Drawing the boxes into the image made it worse (6 → 4 of 16 with pose text).
  ([E02](experiments/E02-ownership-chain/))
- Fine-tuning the 8.31M-parameter readout fitted 30 of 32 training photos but got both targets right on only 6 of 32
  new photos; nearest-torso geometry got 24. A compact local scorer matched geometry (24/32), with identical decisions
  when the photo was removed. ([E02](experiments/E02-ownership-chain/))
- Small decoders on Laya feature grids placed 17.6–19.7% of visible wrists/ankles within 5% of the image diagonal and
  rarely placed both people's joints correctly; a pretrained pose tool placed 98.3% (on COCO photos it was trained on).
  ([E03](experiments/E03-spatial-readouts/))
- Laya Vision's vision tower is value-for-value identical to public SmolVLM-256M-Instruct; its connector changed (and,
  as checked later, its language model). Swapping connectors moved joint localisation by 0.80 points.
  ([E04](experiments/E04-component-audit/))
- Movement of named joints, from Laya grids or from the pose tool, was no better than predicting no movement.
  Averaging left and right into a limb-pair midpoint cut the error by 44% on development pairs.
  ([E05](experiments/E05-limb-pair-movement/))
- On 96 fresh pairs the limb-pair measure passed all four pre-set criteria: error 0.62% vs 1.63% of the image
  diagonal, 258 of 262 answers accepted at 8.5% error (15.5% on the 58 directional ones).
  ([E05](experiments/E05-limb-pair-movement/))

## 2026-09-24
- A 68,280-parameter temporal adapter lost to plain interpolation in two fixed runs; validation selected step 0 both
  times. ([E06](experiments/E06-temporal-completion/))
- More training rows from correlated, mostly upright sources broke a posture readout (17/26 → 11/26, all upright).
  Independent source families matter more than row count. ([E07](experiments/E07-fight-pilots/))
- DINOv2 features beat Laya's under family holdout (0.737 vs 0.582 macro recall), while Laya won on the bus cohort
  (0.797 vs 0.597). ([E07](experiments/E07-fight-pilots/))
- Learned temporal heads lost to interpolating neighbouring frames (0.615 and 0.484 vs 0.825 macro recall); stable
  states inflate hidden-frame scores (75/84 overall, 5/9 next to a posture change). ([E07](experiments/E07-fight-pilots/))
- The frozen DINOv2 posture readout passed a fresh three-family audit: 36/47, macro recall 0.789 against 0.25 for the
  majority class; wrestling stayed at 0.40. ([E07](experiments/E07-fight-pilots/))

## By 2026-09-28 (exact run dates not recorded)
- Under identical crops and head, DINOv2-small reached 82.0% posture accuracy, the Qwen3.8 tower 75.6% and Laya's
  (SmolVLM) tower 70.9%; Qwen had the best pooled macro recall (74.2%) and DINOv2 was fastest (14.4 ms per crop).
  ([E08](experiments/E08-encoder-comparison/))
- Fusing DINOv2 and Qwen raised sparse actor overlap (45.6% → 57.1%) but cut reviewed contact ownership
  (60.5% → 31.3%). ([E08](experiments/E08-encoder-comparison/))

## 2026-10-26 (while preparing this repository)
- Laya Vision's language model also differs from SmolVLM's in all 272 tensors (relative L2 difference 0.85%); the
  published audit script reproduced the original vision-tower and connector results exactly.
  ([E04](experiments/E04-component-audit/))
