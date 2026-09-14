# pan_deng/

Individual contribution folder for Pan Deng — labelled quantitative evaluation of
this project's detectors. Everything under here is additive: it reads the shared
`engines.py` and the `yoloe-*.pt` weights at the repo root, but doesn't modify any
other teammate's files.

## Contents

- `download_coco_subset.py` — pulls a small, labelled COCO val2017 subset (about 240
  images covering ten demo-relevant classes) into `coco_eval_data/` (gitignored;
  regenerate locally rather than pulling from Git).
- `evaluate_detectors.py` — scores YOLOE-26L, YOLOE-26S and Grounding DINO Tiny against
  that subset's real ground-truth boxes with `pycocotools.cocoeval.COCOeval`, producing
  the first labelled accuracy numbers (mAP, not just box counts/latency) for this
  project. Writes to `coco_eval_results/`.
- `coco_eval_results/` — the committed report, per-model raw detections, and a
  `README.md` with the method and results table.
- `requirements-eval.txt` — installs the base app, the Grounding DINO extra, and
  `pycocotools` in one step.

## Quick start

~~~powershell
cd object-detection
python -m pip install -r pan_deng\requirements-eval.txt
python pan_deng\download_coco_subset.py --annotations path\to\instances_val2017.json
python pan_deng\evaluate_detectors.py
~~~

`instances_val2017.json` comes from the official
[COCO 2017 annotations download](https://cocodataset.org/#download)
(`annotations_trainval2017.zip`); only that one file needs extracting.

See `coco_eval_results/README.md` for what the numbers actually show.
