"""Tiny serial precision smoke check; default plan, --run required for models."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tools import run_additional_experiments as experiments


def precision_snapshot(engine, bits):
    """Read the actual inference backend, not the unfused source checkpoint."""
    predictor = engine.model.predictor
    auto = predictor.model
    backend = auto.backend
    native = backend.model
    floating = [p for p in native.parameters() if p.is_floating_point()]
    dtypes = dict(Counter(str(p.dtype) for p in floating))
    expected = 'torch.float16' if bits == 16 else 'torch.float32'
    if not floating or set(dtypes) != {expected} or bool(backend.fp16) != (bits == 16):
        raise RuntimeError(f'Runtime precision mismatch: requested {bits}, observed {dtypes}')
    devices = sorted({str(p.device) for p in floating})
    if not devices or any(not item.startswith('cuda') for item in devices):
        raise RuntimeError(f'Precision GPU smoke unexpectedly ran on {devices}')
    return {'requested_bits': bits, 'actual_parameter_dtype_counts': dtypes,
            'backend_fp16': bool(backend.fp16), 'backend_class': type(backend).__name__,
            'device': devices, 'predictor_quantize': getattr(predictor.args, 'quantize', None),
            'source_parameter_dtype': str(next(engine.model.model.parameters()).dtype),
            'precision_confirmed': True,
            'note': 'Inference backend precision only; source model and visual-reference encoding may remain FP32.'}


def main(argv=None):
    parser = experiments.build_parser()
    parser.description = '1–2 image CUDA runtime precision smoke; no accuracy conclusions.'
    parser.set_defaults(device='CUDA', limit=1, output=Path('work/additional_experiments/gpu_precision_smoke_20261005'),
                        scenarios=['text_baseline','text_fp16','visual_bbox','visual_fp16'])
    parser.add_argument('--run', action='store_true', help='Explicitly run tiny serial GPU smoke after resource checks')
    args = parser.parse_args(argv)
    if args.resume:
        raise ValueError('Runtime dtype smoke requires fresh execution; choose a new output instead of --resume')
    if args.device != 'CUDA' or args.limit not in (1,2):
        raise ValueError('Smoke is limited to CUDA and 1–2 images')
    if set(args.scenarios) - {'text_baseline','text_fp16','visual_bbox','visual_fp16'}:
        raise ValueError('Smoke only accepts the four precision-control scenarios')
    if args.threads > 2 or args.max_det > 100:
        raise ValueError('Smoke allows at most 2 threads and 100 detections')
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      OPENBLAS_NUM_THREADS=str(args.threads))
    os.environ.setdefault('YOLO_CONFIG_DIR',str(PROJECT/'.ultralytics'))
    dataset = experiments.load_dataset(args.dataset,args.limit)
    masks = experiments.mask_provenance(args,dataset)
    configs = [experiments.scenario_config(name,dataset,args) for name in args.scenarios]
    manifest = experiments.manifest_for(args,dataset,masks)
    manifest['runtime_smoke_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.plan or not args.run:
        print(json.dumps({'status':'planned_not_executed','manifest':manifest,'scenarios':configs},ensure_ascii=False,indent=2))
        return 0
    from resource_safety import preflight_inference
    for image in dataset['images']:
        preflight_inference('cpu',640,(int(image['height']),int(image['width']),3))
    import cv2
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(args.threads)
    store = experiments.CheckpointStore(args.output.resolve(),manifest,args.resume)
    report = {'status':'running','scope':'runtime_precision_smoke_not_accuracy_evaluation',
              'image_count':len(dataset['images']),'manifest_fingerprint':store.fingerprint,
              'precision_audits':{},'scenarios':{}}
    audit_file=store.output/'runtime_dtype.json'
    def factory(config,arguments):
        engine=experiments.load_engine(config,arguments)
        original=engine.detect
        def monitored(*values,**options):
            result=original(*values,**options)
            report['precision_audits'][config['name']]=precision_snapshot(engine,config['inference_options']['quantize'])
            experiments.atomic_json(audit_file,report['precision_audits'])
            return result
        engine.detect=monitored
        return engine
    try:
        for config in configs:
            report['scenarios'][config['name']]=experiments.run_scenario(config,dataset,args,store,masks,engine_factory=factory)
        if set(report['precision_audits']) != set(args.scenarios):
            raise RuntimeError('Not every requested scenario has actual runtime precision evidence')
        report['status']='runtime_precision_confirmed'
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        experiments.atomic_json(store.output/'smoke_report.json',report)
        raise
    experiments.atomic_json(store.output/'smoke_report.json',report)
    print(json.dumps({'status':report['status'],'precision':report['precision_audits']},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
