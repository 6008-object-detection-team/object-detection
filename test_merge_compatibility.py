"""Keep the earlier token-scored DINO evaluation API during the Windows upgrade."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from engines import GroundingDinoEngine, GroundingDINOEngine


class LegacyDinoTests(unittest.TestCase):
    def test_legacy_token_scoring_and_box_cap_survive_merge(self):
        import torch
        engine = GroundingDinoEngine()
        self.assertIsInstance(engine, GroundingDINOEngine)
        engine.processor = MagicMock()
        engine.processor.return_value = {"input_ids": torch.tensor([[101, 1, 2, 102]])}
        engine.processor.tokenizer.return_value = {
            "offset_mapping": [(0, 0), (0, 6), (8, 14), (0, 0)]
        }
        probabilities = torch.tensor([[[0.01, 0.8, 0.2, 0.01], [0.01, 0.1, 0.7, 0.01]]])
        engine.model = MagicMock(return_value=SimpleNamespace(
            logits=torch.logit(probabilities),
            pred_boxes=torch.tensor([[[0.2, 0.4, 0.2, 0.3], [0.8, 0.4, 0.1, 0.2]]]),
        ))
        frame = np.zeros((100, 200, 3), np.uint8)
        with patch("engines.preflight_inference") as preflight:
            low = engine.detect(frame, 0.05, class_names=["person", "bottle"])
            high = engine.detect(frame, 0.4, class_names=["person", "bottle"])
            capped = engine.detect(frame, 0.05, class_names=["person", "bottle"], max_det=1)
        self.assertEqual([(d.label, d.class_id) for d in low], [("person", 0), ("bottle", 1)])
        self.assertEqual([(d.label, d.class_id) for d in low], [(d.label, d.class_id) for d in high])
        self.assertEqual(len(capped), 1)
        self.assertEqual(preflight.call_count, 3)

    def test_legacy_prompts_keep_correct_token_spans(self):
        prompt, spans = GroundingDinoEngine._build_prompt(["person", "mobile phone"])
        self.assertEqual([prompt[a:b] for a, b in spans], ["person", "mobile phone"])
        self.assertEqual(GroundingDinoEngine._class_token_groups(
            [(0, 0), (0, 6), (6, 7), (8, 14), (15, 20), (20, 21)], spans
        ), [[1], [3, 4]])


if __name__ == "__main__":
    unittest.main()
