"""Visual prompt geometry, cache/state and resource safety without GPU work."""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from engines import OpenVocabEngine
from visual_engine import VisualPromptEngine, prepare_reference


class ResourceLimitError(RuntimeError):
    pass


def fake_safety():
    return types.SimpleNamespace(
        ResourceLimitError=ResourceLimitError,
        preflight_inference=MagicMock(),
        handle_cuda_oom=MagicMock(return_value=False),
    )


class FakeEmbedding:
    def __init__(self):
        self.moves = []

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self

    def to(self, **kwargs):
        self.moves.append(kwargs)
        return self


class ReferenceGeometryTests(unittest.TestCase):
    def test_whole_image_is_letterboxed_and_box_maps_with_padding(self):
        image = np.full((20, 40, 3), (5, 80, 160), dtype=np.uint8)
        original = image.copy()
        padded, box = prepare_reference(image, (10, 5, 30, 15), 64)
        self.assertEqual(padded.shape, (64, 64, 3))
        np.testing.assert_allclose(box, [[16, 24, 48, 40]])
        np.testing.assert_array_equal(padded[0, 0], [114, 114, 114])
        np.testing.assert_array_equal(padded[16, 0], image[0, 0])
        np.testing.assert_array_equal(image, original)

    def test_portrait_and_odd_padding_use_actual_left_and_top(self):
        image = np.zeros((40, 21, 3), dtype=np.uint8)
        _, box = prepare_reference(image, (0, 0, 21, 40), 64)
        np.testing.assert_allclose(box, [[15, 0, 48.6, 64]])

    def test_invalid_image_boxes_and_oversized_inputs_fail_before_cuda(self):
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        for box in [(-1, 0, 10, 10), (1, 2, 1, 5), (0, 0, 50, 10),
                    (0, 0, np.nan, 1), (0, 0, 2)]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                prepare_reference(frame, box, 64)
        for size in [True, 31, 961, 64.5]:
            with self.subTest(imgsz=size), self.assertRaises(ValueError):
                prepare_reference(frame, (0, 0, 10, 10), size)
        with self.assertRaises(ValueError):
            prepare_reference(frame.astype(np.float32), (0, 0, 10, 10), 64)


class VisualStateTests(unittest.TestCase):
    def setUp(self):
        self.engine = VisualPromptEngine()
        self.engine.imgsz = 64
        self.engine.model = MagicMock()
        self.engine.model.model.parameters.return_value = iter([
            types.SimpleNamespace(device="cpu", dtype="float32"),
        ])
        self.reference = np.zeros((20, 40, 3), dtype=np.uint8)
        self.box = (5, 5, 15, 15)
        self.embedding = FakeEmbedding()
        self.safety = fake_safety()
        self.module_patch = patch.dict(sys.modules, {"resource_safety": self.safety})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_prompt_encoded_once_and_installed_explicitly_without_text(self):
        with patch.object(self.engine, "_encode_visual_prompt", return_value=self.embedding) as encode:
            for _ in range(3):
                self.assertEqual(self.engine.set_visual_prompt(self.reference, self.box, "bottle"), "SUCCESS")
        encode.assert_called_once()
        self.engine.model.set_classes.assert_called_once_with(["bottle"], self.embedding)
        self.assertEqual(self.engine.reference_shape, self.reference.shape)
        self.assertEqual(self.engine.reference_box, self.box)
        self.assertEqual(self.engine.current_classes, ["bottle"])
        self.engine.model.get_text_pe.assert_not_called()
        self.safety.preflight_inference.assert_called_with("cpu", 64, self.reference.shape)

    def test_changed_reference_reencodes_but_label_only_rename_reuses_embedding(self):
        with patch.object(self.engine, "_encode_visual_prompt", return_value=self.embedding) as encode:
            self.engine.set_visual_prompt(self.reference, self.box, "bottle")
            self.engine.set_visual_prompt(self.reference, self.box, "reference")
            self.assertEqual(encode.call_count, 1)
            changed = self.reference.copy()
            # Outside the ROI also matters because SAVPE uses the whole image.
            changed[0, 0, 0] = 255
            self.engine.set_visual_prompt(changed, self.box, "reference")
            self.assertEqual(encode.call_count, 2)

    def test_detect_preserves_visual_embedding_and_forwards_options(self):
        with patch.object(self.engine, "_encode_visual_prompt", return_value=self.embedding):
            self.engine.set_visual_prompt(self.reference, self.box, "bottle")
        with patch.object(OpenVocabEngine, "detect", return_value=[]) as detect:
            self.engine.detect(self.reference, 0.12, 0.35, ["person"], "tile", 17)
        detect.assert_called_once_with(self.reference, 0.12, 0.35, ["bottle"], "tile", 17)
        self.engine.model.set_classes.assert_called_once()

    def test_release_forces_text_reencoding_even_when_category_name_matches(self):
        with patch.object(self.engine, "_encode_visual_prompt", return_value=self.embedding):
            self.engine.set_visual_prompt(self.reference, self.box, "bottle")
        self.engine.model.model.names = {0: "bottle"}
        self.engine.model.model.pe = self.embedding
        self.engine.release_visual_prompt()
        self.assertEqual(self.engine.model.model.names, {})
        self.assertFalse(hasattr(self.engine.model.model, "pe"))
        self.assertIsNone(self.engine.model.predictor)
        self.assertIsNone(self.engine.visual_embedding)
        self.assertEqual(self.engine.current_classes, [])
        with self.assertRaisesRegex(RuntimeError, "工作线程"):
            self.engine.detect(self.reference)

    def test_oom_stops_encoding_without_retry_or_fallback(self):
        self.safety.handle_cuda_oom.return_value = True
        with patch.object(self.engine, "_encode_visual_prompt", side_effect=RuntimeError("CUDA out of memory")) as encode:
            with self.assertRaisesRegex(ResourceLimitError, "显存不足停止"):
                self.engine.set_visual_prompt(self.reference, self.box)
        self.assertEqual(encode.call_count, 1)
        self.engine.model.set_classes.assert_not_called()
        self.assertIsNone(self.engine.visual_embedding)

    def test_preflight_failure_does_not_start_embedding_or_model_installation(self):
        self.safety.preflight_inference.side_effect = ResourceLimitError("memory budget")
        with patch.object(self.engine, "_encode_visual_prompt") as encode:
            with self.assertRaises(ResourceLimitError):
                self.engine.set_visual_prompt(self.reference, self.box)
        encode.assert_not_called()
        self.engine.model.set_classes.assert_not_called()

    def test_drawing_ignores_legacy_colour_filter(self):
        with patch.object(OpenVocabEngine, "draw_detections", return_value=self.reference) as draw:
            self.engine.draw_detections(self.reference, [], np.zeros((5, 5, 3), dtype=np.uint8), self.reference)
        draw.assert_called_once_with(self.reference, [], None, self.reference)


if __name__ == "__main__":
    unittest.main()
