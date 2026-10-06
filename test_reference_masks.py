"""Reference masks preserve geometry and invalidate cached visual prompts."""
import unittest
import sys
from unittest.mock import MagicMock, patch

import numpy as np

from test_visual_engine import FakeEmbedding, fake_safety
from visual_engine import VisualPromptEngine, prepare_reference_mask


class ReferenceMaskGeometryTests(unittest.TestCase):
    def test_padding_is_background_and_full_foreground_is_binary(self):
        source = np.full((20, 40), 255, np.uint8)
        original = source.copy()
        mapped = prepare_reference_mask(source, (20, 40, 3), 64)
        self.assertEqual(mapped.shape, (64, 64))
        self.assertTrue((mapped[:16] == 0).all())
        self.assertTrue((mapped[16:48] == 1).all())
        self.assertTrue((mapped[48:] == 0).all())
        np.testing.assert_array_equal(source, original)

    def test_odd_padding_matches_image_letterbox(self):
        mapped = prepare_reference_mask(np.ones((40, 21), bool), (40, 21, 3), 64)
        self.assertTrue((mapped[:, :15] == 0).all())
        self.assertTrue((mapped[:, 15:49] == 1).all())
        self.assertTrue((mapped[:, 49:] == 0).all())

    def test_invalid_masks_and_unsafe_sizes_fail(self):
        for source in [np.zeros((20, 40), np.uint8), np.ones((21, 40), np.uint8),
                       np.ones((20, 40), np.float32), np.full((20, 40), 7, np.uint8)]:
            with self.subTest(shape=source.shape, dtype=source.dtype), self.assertRaises(ValueError):
                prepare_reference_mask(source, (20, 40, 3), 64)
        for size in (True, 31, 961, 64.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                prepare_reference_mask(np.ones((20, 40), np.uint8), (20, 40, 3), size)


class ReferenceMaskCacheTests(unittest.TestCase):
    def setUp(self):
        self.engine = VisualPromptEngine()
        self.engine.imgsz = 64
        self.engine.model = MagicMock()
        self.reference = np.zeros((20, 40, 3), dtype=np.uint8)
        self.box = (5, 5, 15, 15)
        self.embedding = FakeEmbedding()
        self.safety = patch.dict(sys.modules, {'resource_safety': fake_safety()})
        self.safety.start()
        self.addCleanup(self.safety.stop)

    def test_mask_change_reencodes_and_equivalent_binary_values_reuse(self):
        first = np.zeros((20, 40), np.uint8)
        first[5:15, 5:15] = 1
        changed = first.copy()
        changed[5, 5] = 0
        with patch.object(self.engine, '_encode_visual_prompt', return_value=self.embedding) as encode:
            self.engine.set_visual_prompt(self.reference, self.box, 'bottle', reference_mask=first)
            self.engine.set_visual_prompt(self.reference, self.box, 'bottle', reference_mask=first * 255)
            self.assertEqual(encode.call_count, 1)
            self.engine.set_visual_prompt(self.reference, self.box, 'bottle', reference_mask=changed)
            self.assertEqual(encode.call_count, 2)
            self.engine.set_visual_prompt(self.reference, self.box, 'bottle')
            self.assertEqual(encode.call_count, 3)


if __name__ == '__main__':
    unittest.main()
