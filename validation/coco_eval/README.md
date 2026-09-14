# Quantitative comparison: YOLOE vs. Grounding DINO Tiny

First labelled accuracy evaluation for this project. Earlier validation runs (see
`../grounding_dino_preview/` and the rest of `../`) checked that things work and how
fast they run; this one scores real detections against real ground-truth boxes.

## Method

- **Data**: 240 images from COCO val2017, selected (round-robin, not just the first N)
  so all ten target classes are represented, with their official instance annotations.
  Not the full 5,000-image split — see `download_coco_subset.py`.
- **Classes**: `person, bicycle, car, dog, backpack, bottle, cup, chair, laptop,
  cell phone` — ten common COCO categories, chosen to overlap with the app's own demo
  prompts. This is *not* a test of either model's full open-vocabulary range, only of
  these ten words.
- **Models**: YOLOE-26L, YOLOE-26S, and Grounding DINO Tiny, each given the same ten
  class names as its text prompt and scored with `pycocotools.cocoeval.COCOeval`
  (`bbox` mode) — the same tool and metric definitions (mAP@[.5:.95], mAP@.5, AR) used
  in standard detection benchmarks, so these numbers are directly comparable to ones
  reported elsewhere.
- **Confidence threshold**: 0.05 for YOLOE (matches how COCOeval expects a
  precision-recall sweep, not a hard operating cutoff). Grounding DINO Tiny was run at
  **both** 0.05 and its own documented default of 0.35 — see Results below for why that
  distinction turned out to matter.
- Run with `evaluate_detectors.py`; full per-detection output kept alongside
  `report.json` for auditability.

## Results

| Model | mAP@[.5:.95] | mAP@.5 | ms/image | Detections |
| --- | ---: | ---: | ---: | ---: |
| YOLOE-26L | **0.473** | 0.702 | 67 | 3,475 |
| YOLOE-26S | 0.418 | 0.638 | 35 | 3,592 |
| Grounding DINO Tiny (threshold 0.05) | 0.132 | 0.176 | 378 | 40,754 |
| Grounding DINO Tiny (threshold 0.35, its own default) | 0.414 | 0.544 | 306 | 2,574 |

## What this shows

- On these ten classes, **YOLOE-26L is the accuracy leader**, ahead of YOLOE-26S by
  5.5 points of mAP@[.5:.95] and roughly twice as fast as Grounding DINO Tiny at either
  threshold.
- **Grounding DINO Tiny is highly sensitive to its confidence threshold.** At 0.05 it
  floods each image with duplicate low-score boxes (170 detections/image on average)
  and its mAP collapses to 0.132 — that is *not* a fair reading of the model. At its
  own documented default (0.35) it produces a plausible number of boxes and its
  mAP@[.5:.95] (0.414) lands close to YOLOE-26S (0.418), i.e. roughly comparable
  accuracy, at about 9× the latency.
- This is why `compare_grounding_dino.py`'s unlabelled preview (box counts and scores
  only, no ground truth) could not have surfaced this: raw box counts alone made
  Grounding DINO look like it was finding *more* objects, when most of the extra boxes
  at a low threshold are false positives.

## Limits

- Ten common classes only, not a test of open-vocabulary phrases like "red backpack"
  or attribute-level queries — COCO's category labels don't support that.
- One fixed IoU=0.45 for NMS on both engines; not swept.
- Single run per configuration, no repeated-seed variance estimate.
- 240 images is a deliberately small, fast-to-reproduce subset, not the full COCO
  val2017 split; per-class AP with this few examples per class is noisier than a
  full-split evaluation.
