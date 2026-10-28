"""Official SAHI slicing + class-aware NMS, reusing one detector serially."""
from dataclasses import replace

import cv2
import numpy as np

from detection_utils import Detection
from resource_safety import ResourceLimitError, preflight_inference
from i18n import message

# Slice sizes offered by the GUI combo (ui.py); used only to word the advice.
GUI_SLICE_SIZES = (320, 512, 640)
# A crop box within this many pixels of an internal crop edge is a truncated object.
EDGE_MARGIN = 3


def slice_count(shape, slice_size, slice_overlap=0.2):
    """Rows x columns of the safety plan (no crops are created)."""
    height, width = shape[:2]
    stride = int(slice_size) - int(slice_size * slice_overlap)
    rows = 1 + max(0, int(np.ceil((height - slice_size) / stride)))
    cols = 1 + max(0, int(np.ceil((width - slice_size) / stride)))
    return rows * cols


def validate_slice_plan(shape, slice_size=512, slice_overlap=0.2, max_slices=9):
    if isinstance(slice_size, bool) or int(slice_size) != slice_size or not 128 <= slice_size <= 960:
        raise ValueError('切片尺寸必须是 128 到 960 之间的整数')
    if not np.isfinite(slice_overlap) or not 0 <= slice_overlap <= 0.3:
        raise ValueError('安全模式切片重叠比例要求在 0 到 0.3 之间')
    if isinstance(max_slices, bool) or int(max_slices) != max_slices or not 1 <= max_slices <= 9:
        raise ValueError('安全模式每张图最多允许 9 个切片')
    height, width = shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('输入图像为空')
    # Bound the work before asking SAHI to materialize any crops.
    count = slice_count(shape, slice_size, slice_overlap)
    if count > max_slices:
        values = dict(width=width, height=height, size=int(slice_size), count=count, maximum=max_slices)
        fits = [size for size in GUI_SLICE_SIZES
                if size > slice_size and slice_count(shape, size, slice_overlap) <= max_slices]
        if fits:
            raise ResourceLimitError(message(
                '本图 {width}×{height} 用 {size} 像素切片需要 {count} 块，超过 {maximum} 块上限；可改用 {fit} 像素切片，或缩小输入图像。',
                fit=fits[0], **values))
        # No offered size fits: "use larger slices" would be wrong advice.
        raise ResourceLimitError(message(
            '本图 {width}×{height} 用 {size} 像素切片需要 {count} 块，超过 {maximum} 块上限；界面提供的切片尺寸都放不下，请先缩小输入图像或关闭 SAHI。',
            **values))
    from sahi.slicing import get_slice_bboxes
    boxes = get_slice_bboxes(height, width, int(slice_size), int(slice_size),
                             auto_slice_resolution=False,
                             overlap_height_ratio=slice_overlap, overlap_width_ratio=slice_overlap)
    if len(boxes) > max_slices:
        raise ResourceLimitError('SAHI 切片数超过安全上限')
    return boxes


def touches_internal_edge(xyxy, left, top, crop_width, crop_height, width, height, margin=EDGE_MARGIN):
    """True when a crop box touches a crop edge that lies inside the image.

    Such a box is a fragment of an object cut by the slice border. The full-image
    pass, or the overlapping neighbour slice that holds the whole object, still
    reports it, so the fragment is dropped before the merge. Boxes on the outer
    image border are kept. Same 3 px rule as OpenVocabEngine.detect_tiles.
    """
    x1, y1, x2, y2 = xyxy
    return ((left > 0 and x1 < margin) or (top > 0 and y1 < margin)
            or (left + crop_width < width and x2 > crop_width - margin)
            or (top + crop_height < height and y2 > crop_height - margin))


def merge_sahi(detections, iou, max_det):
    from sahi.prediction import ObjectPrediction
    from sahi.postprocess.combine import NMSPostprocess
    objects = []
    for detection in detections:
        prediction = ObjectPrediction(bbox=list(detection.xyxy), score=detection.score,
                                      category_id=detection.class_id, category_name=detection.label)
        prediction.detection_record = detection
        objects.append(prediction)
    if len(objects) > 1:
        objects = NMSPostprocess(match_threshold=iou, match_metric='IOU', class_agnostic=False)(objects)
    return [item.detection_record for item in sorted(objects, key=lambda item: item.score.value, reverse=True)[:max_det]]


def detect_sliced(engine, frame, conf=0.4, iou=0.45, class_names=None, max_det=300,
                  slice_size=512, slice_overlap=0.2, max_slices=9, base_detections=None):
    from engines import validate_detection_options
    max_det = validate_detection_options(conf, iou, max_det)
    preflight_inference(engine.device, engine.imgsz, frame.shape)
    boxes = validate_slice_plan(frame.shape, slice_size, slice_overlap, max_slices)
    base = list(base_detections) if base_detections is not None else engine.detect(
        frame, conf, iou, class_names, max_det=max_det)
    if len(boxes) == 1:
        return merge_sahi(base, iou, max_det)
    from PIL import Image
    from sahi.slicing import slice_image
    sliced = slice_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                         slice_height=int(slice_size), slice_width=int(slice_size),
                         overlap_height_ratio=slice_overlap, overlap_width_ratio=slice_overlap,
                         auto_slice_resolution=False, verbose=False)
    candidates = base
    height, width = frame.shape[:2]
    # SAHI supplies RGB images and global offsets. Only one crop is inferred at a time.
    for image, (left, top) in zip(sliced.images, sliced.starting_pixels):
        crop = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        crop_height, crop_width = crop.shape[:2]
        for detection in engine.detect(crop, conf, iou, class_names, source='slice', max_det=max_det):
            # Cut-off fragments would survive IoU-NMS next to the whole object (low IoU).
            if touches_internal_edge(detection.xyxy, left, top, crop_width, crop_height, width, height):
                continue
            x1, y1, x2, y2 = detection.xyxy
            candidates.append(replace(detection, xyxy=(x1 + left, y1 + top, x2 + left, y2 + top)))
    return merge_sahi(candidates, iou, max_det)
