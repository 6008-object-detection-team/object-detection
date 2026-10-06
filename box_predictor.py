"""Box-only postprocessing for YOLOE segmentation checkpoints.

The application evaluates and draws boxes. Constructing one full-resolution
mask per detection needlessly consumes memory, particularly during SAHI or on
high-resolution camera frames. This predictor retains segmentation head class
counts for NMS while never reconstructing prototype masks.
"""
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG


class BoxOnlySegmentationPredictor(DetectionPredictor):
    """Read segmentation boxes without allocating instance mask images."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # Segmentation output includes mask coefficients after class scores.
        # DetectionPredictor must pass len(names), rather than nc=0, to NMS;
        # otherwise those coefficients could be interpreted as extra classes.
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        # Same outer prediction selection as the official SegmentationPredictor.
        # DetectionPredictor/NMS already supports a nested PyTorch output tuple.
        # Deliberately omit protos and inherit box-only Results construction.
        kwargs.pop("protos", None)
        return super().postprocess(preds[0], img, orig_imgs, **kwargs)
