# Where the numbers come from

Every number in this repository is copied from a saved result report of the study; none is recomputed, except the
one row marked *this repository*. Paths are relative to the `Laya Vision/` folder of Dhruva's research archive (not
public); `RS` is that folder's `RESEARCH_SUMMARY.md` (compiled 2026-09-28). Line numbers refer to the archived files.

## README and FINDINGS headline
| number | source |
|---|---|
| study conclusion ("has not established reliable general human-state understanding ...") | RS:9 |
| 6.18 s per four-frame question on CPU, 1.16 s with cached features | `research/RESULTS.md`:104; RS:54 |
| 14.4 / 23.4 / 79.4 ms per crop (DINOv2 / Laya tower / Qwen tower) | `encoder_comparison/RESULTS.md`:11-13 |
| ownership size curve prepared, not run | RS:107 |
| 34 planned label requests, 5 saved, 1 invalid; 74 settled calls; $13.106700 conservative | RS:185 |
| laptop GPU (RTX 3060), no paid compute in the early studies | RS:242 |

## E00 pilot
| number | source |
|---|---|
| 528 questions, 95 windows, 263 images, 29 groups; 454 agreed; 72 disagreed | `research/RESULTS.md`:19 |
| test 92 questions, 36 posture, 5 groups | `research/RESULTS.md`:21 |
| 36.1 / 36.1 / 50.0 / 50.0 / 66.7 / 38.9% and macro-F1 values | `research/RESULTS.md`:33-40 |
| +0.0045 macro-F1 vs +0.05; same 18/36 | `research/RESULTS.md`:42 |
| 91.7% = no-image prior (11/12 upright); no seated/reclined predictions | `research/RESULTS.md`:46 |
| probe standardisation amendment | `research/RESULTS.md`:52 |
| 29.4% / 33.3% test error at a 10% calibration target | `research/RESULTS.md`:58 |
| UT-Interaction 80.0 / 75.0 / 68.3% | `research/RESULTS.md`:90-92 |

## E01 controls
| number | source |
|---|---|
| 468 control queries per condition | `research/RESULTS.md`:66 |
| option order 39/92 (42.4%), target 90% | `research/RESULTS.md`:70 |
| actor swap 1/19 (5.3%), target 80% | `research/RESULTS.md`:71 |
| blank 72/92 unchanged incl. 36/41 correct; shuffled 39/46; ROI 31/35 | `research/RESULTS.md`:72 |
| four frames vs last frame identical on 8 questions | `research/RESULTS.md`:74 |
| directional reversal 0/7 | `research/RESULTS.md`:78 |
| tokenizer audit 92 pairs / 848 markers; 41/92 → 45/92; posture 18/36 | `research/RESULTS.md`:82 |
| video: 6/8 combinations change with option order; blank never "insufficient evidence" | RS:120 |
| video: 768 decisions; same winner on 31/32 slots; blank never "unknown" | RS:121 |

## E02 ownership chain
| number | source |
|---|---|
| 172 questions, 96 photo groups | `ownership_reference/RESULTS.md`:9 |
| learned geometry 125/172 (72.7%); visual heads 90/172 (52.3%) | `ownership_reference/RESULTS.md`:17,23-26 |
| heads selected step zero | `ownership_reference/RESULTS.md`:32 |
| AI teacher 40/40; geometry 27/40 | `ownership_reference/RESULTS.md`:42 |
| exact binomial interval 91.19–100% | `ownership_reference/RESULTS.md`:46 |
| visual heads unchanged under blank / shuffle / wrong crop | `ownership_reference/RESULTS.md`:52 |
| 16 photos: geometry 13/16 (+3 unknown); Laya 5/16; blank + pose 7/16 | `ownership_repair/RESULTS.md`:3,17-21 |
| all six orders: 6/16; B on 15/16; B wins 85/96 and blank 96/96 | `ownership_order/RESULTS.md`:3,13-17 |
| renaming consistency 0/16 | `ownership_order/RESULTS.md`:24 |
| markers: image-only 5/16, pose 4/16, blank 7/16; renaming 5/16 | `ownership_markers/RESULTS.md`:3,9-13,28 |
| readout 8,313,409 parameters; fitted 14/16; blank 8/16 | `ownership_supervised/RESULTS.md`:9,15-18 |
| renaming consistency 12/16 after fitting | `ownership_supervised/RESULTS.md`:30 |
| pairs: train 30/32, development 6/32; geometry 24/32; fitted blank 5/32 | `ownership_pairs/RESULTS.md`:3,7-11 |
| locality scorer 24/32; blank 24/32; all 64 decisions identical | `ownership_locality/RESULTS.md`:3,24,30-31 |
| question construction rules | `ownership_reference/dataset.py` (ported to `src/ownership_questions.py`; same 36 questions on COCO val2017) |
| demo figure answers (q000, q001, q005) | `ownership_order/evaluation-v1.json`, `six_order.per_query` |

## E03 spatial readouts
| number | source |
|---|---|
| 346 / 86 photos, 2,315 / 598 observations | `person_grounding/RESULTS.md`:9 |
| 36 / 2,340 / 6,948 parameters | `person_grounding/RESULTS.md`:13 |
| 17.56 / 19.06 / 19.73% PCK; identity 39.5 / 30.0 / 32.0%; both localised 4.0 / 2.0 / 2.0% | `person_grounding/RESULTS.md`:25-30 |
| reference minus local +0.6689, interval −1.9705 to +3.1097 | `person_grounding/RESULTS.md`:3 |
| peak decoding 139/598 (23.24%); identity 26.5%; both localised 4.0% | `heatmap_diagnostics/RESULTS.md`:3,15,25-26 |
| 16 photos: 13 / 18 / 23 / 114 of 116; pairs 4 / 1 / 1 / 42 of 44 | `pose_reference_pilot/RESULTS.md`:13-16,24-27 |
| COCO train2017 exposure caveat | `pose_reference_pilot/RESULTS.md`:5 |

## E04 component audit
| number | source |
|---|---|
| SmolVLM revision 7e3e67e…, not a verified training parent | `temporal_reference/parent_identity/IDENTITY_AUDIT.md`:8 |
| vision 197 tensors, 86,433,024 values, 0 different | `temporal_reference/parent_identity/IDENTITY_AUDIT.md`:20 |
| connector 7,077,888 values, 7,077,887 different, relative L2 0.068 | `temporal_reference/parent_identity/IDENTITY_AUDIT.md`:21 |
| language model 272 tensors, 134,588,736 values, 134,588,392 different, relative L2 0.0085 | *this repository*: `src/component_audit.py`, run 2026-10-26 |
| connector comparison 925 vs 908 of 2,132 (+0.7974, interval 0 to +1.5712) | `connector_comparison/RESULTS.md`:3,21 |
| movement accuracy 43.7 vs 42.8% | `connector_comparison/RESULTS.md`:24 |

## E05 movement
| number | source |
|---|---|
| Laya-grid decoder +11.67 points; direct classifier at initialisation; no policy qualified | `temporal_reference/RESULTS.md`:3,35,56 |
| axis error .03758 vs zero .02127 | `temporal_reference/RESULTS.md`:43,46 |
| opposite-side joint closer for 43.21% | `temporal_reference/RESULTS.md`:50 |
| pose tool: 55.49% balanced accuracy; 23.86% error at 66.0% coverage | `pose_transfer/RESULTS.md`:3,42 |
| pose tool axis error .02180 vs same-support zero .02095 | `pose_transfer/RESULTS.md`:27-28 |
| side audit 4/64 slots | RS:82 |
| development limb pairs .005644 vs .010108 (44.2%); 37/40 | `centroid_diagnostics/RESULTS.md`:3,21-23 |
| confirmation 258/262 accepted, 22 errors (8.53%); .006189 vs .016313 (62.06%) | `centroid_transfer/RESULTS.md`:3,27,39 |
| 131/192 valid slots; 202/262 little change; 9 of 58 directional (15.52%) | `centroid_transfer/RESULTS.md`:5 |

## E06 temporal completion
| number | source |
|---|---|
| 68,280 parameters; inputs | `temporal_completion/RESULTS.md`:9 |
| 3,072 / 96 / 96 / 192 windows; reused roles | `temporal_completion/RESULTS.md`:11 |
| step 0 selected; 1.5% and 4.7% worse | `temporal_completion/RESULTS.md`:15 |
| errors 0.461 / 0.187 / 0.238 etc. | `temporal_completion/RESULTS.md`:21-25 |
| 133 demo samples | `temporal_completion/RESULTS.md`:37 |

## E07 fighting pilots
| number | source |
|---|---|
| bus development 17/26 vs majority 11/26 | `fight_experiment/spatial-v1/RESULTS.md`:5-9 |
| refit: 82 added crops; 11/26, all upright, equal to blank | `fight_experiment/spatial-v2/RESULTS.md`:3,5 |
| three audits, 52 targets, .325926 vs .333333 | RS:164 |
| coverage: DINOv2 .7368 vs Laya .5824; bus .5970 vs .7970; 184 rows | `fight_experiment/pilot2-coverage/RESULTS.md`:3,7-11 |
| hidden centre .8247 / .7579 / .6151 / .4835 / .2130 | `fight_experiment/pilot2-temporal/RESULTS.md`:7-11 |
| 5/9 next to a change vs 75/84 overall | `fight_experiment/pilot2-temporal/RESULTS.md`:17 |
| fresh audit 36/47, .7892; majority .25; geometry .3599; shuffled .2353 | `fight_experiment/pilot2-audit/RESULTS.md`:3 |
| wrestling .40; Laya .2407 | `fight_experiment/pilot2-audit/RESULTS.md`:7,11 |
| partner crop .40 → .10 on 19 targets | `fight_experiment/pilot2-audit/RESULTS.md`:13 |
| gate: 52/58 repeat agreements, 64/66 attribution | RS:177 |
| upload: 16 frames, 64 predictions | RS:179 |
| data licences per source | `fight_experiment/dataset-v*/initial-source-roles.json`; `fight_experiment/SOURCE_CATALOG.md` |

## E08 encoder comparison
| number | source |
|---|---|
| 245 crops, 15 videos, 172 known, 184 training rows | `encoder_comparison/RESULTS.md`:7 |
| 82.0 / 70.9 / 75.6% accuracy; macro recalls; ms per crop | `encoder_comparison/RESULTS.md`:11-13 |
| 15/18 seated, 20/30 crouched; 99/102 upright, 21/22 lying | `encoder_comparison/RESULTS.md`:15 |
| 260-parameter head, PCA 64 | `encoder_comparison/RESULTS.md`:35 |
| four clips, 96 frames, 28 scored | `spatial_encoder_comparison/RESULTS.md`:7 |
| IoU 45.6 / 45.5 / 57.1%; macro recall; contact 60.5 / 50.2 / 31.3% | `spatial_encoder_comparison/RESULTS.md`:13-15 |
| spatial-removal control 3.1 / 6.0 / 3.1% | `spatial_encoder_comparison/RESULTS.md`:34-36 |
| 24,738 parameters; 36 fits | `spatial_encoder_comparison/RESULTS.md`:44 |
