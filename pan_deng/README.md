# pan_deng/

Individual contribution folder for Pan Deng — labelled quantitative evaluation of
this project's detectors. It reads the shared `engines.py` and the `yoloe-*.pt`
weights at the repo root. The one change outside this folder is a bug fix in
`GroundingDinoEngine.detect` (class labels no longer depend on the confidence
threshold); see `coco_eval_results/README.md`, "Correction".

## Contents

- `download_coco_subset.py` — pulls a small, labelled COCO val2017 subset (about 240
  images covering ten demo-relevant classes) into `coco_eval_data/` (gitignored;
  regenerate locally rather than pulling from Git).
- `evaluate_detectors.py` — scores YOLOE-26L, YOLOE-26S and Grounding DINO Tiny against
  that subset's real ground-truth boxes with `pycocotools.cocoeval.COCOeval`, producing
  the first labelled accuracy numbers (mAP, not just box counts/latency) for this
  project. Writes to `coco_eval_results/`.
- `sweep_threshold_topk.py` — runs each model once, then tests the confidence
  threshold and the number of boxes kept per image separately, offline. Writes
  `coco_eval_results/sweep/sweep.json`; raw detections are cached in
  `coco_eval_data/sweep_cache/`.
- `plot_comparison.py`, `plot_sweep.py` — render the charts from those JSON files.
- `coco_eval_results/` — the committed report, per-model raw detections, the sweep
  results, and a `README.md` with the method and results tables.
  `legacy_text_threshold/` keeps the pre-fix Grounding DINO results for comparison.
- `requirements-eval.txt` — installs the base app, the Grounding DINO extra, and
  `pycocotools` in one step.

## Quick start

~~~powershell
cd object-detection
python -m pip install -r pan_deng\requirements-eval.txt
python pan_deng\download_coco_subset.py --annotations path\to\instances_val2017.json
python pan_deng\evaluate_detectors.py
python pan_deng\sweep_threshold_topk.py
python pan_deng\plot_comparison.py
python pan_deng\plot_sweep.py
~~~

`instances_val2017.json` comes from the official
[COCO 2017 annotations download](https://cocodataset.org/#download)
(`annotations_trainval2017.zip`); only that one file needs extracting.

See `coco_eval_results/README.md` for what the numbers actually show.
