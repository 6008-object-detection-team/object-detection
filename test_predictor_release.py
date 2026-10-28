"""Every prompt or mode change frees the predictor it dropped, before the next
one is built. Fake models with a reference cycle, no GPU work."""
import gc
import os
import types
import unittest
import weakref
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from engines import OpenVocabEngine
from main import AIApp
from visual_engine import VisualPromptEngine


class CyclicPredictor:
    """Like Ultralytics' predictor, only the cyclic collector can free it."""

    def __init__(self):
        self.cycle = self


class FakeYOLOE:
    def __init__(self):
        self.predictor = CyclicPredictor()
        self.encoded = []
        self.model = types.SimpleNamespace(names={})

    def set_classes(self, classes, embeddings=None):
        # YOLOE.set_classes drops the predictor after installing the prompt.
        self.encoded.append(list(classes))
        self.predictor = None


class AutomaticGcDisabled:
    """Only explicit collections may free cycles while a test checks one."""

    def __enter__(self):
        gc.collect()
        self.enabled = gc.isenabled()
        gc.disable()

    def __exit__(self, *exc):
        if self.enabled:
            gc.enable()


class PredictorReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_class_change_frees_the_dropped_predictor_before_the_next_frame(self):
        engine = OpenVocabEngine()
        engine.model = FakeYOLOE()
        with AutomaticGcDisabled():
            dropped = weakref.ref(engine.model.predictor)
            engine._configure_classes(["cup"])
            self.assertIsNone(dropped())
            # Unchanged classes neither re-encode nor collect.
            engine.model.predictor = CyclicPredictor()
            kept = weakref.ref(engine.model.predictor)
            with patch("gc.collect") as collect:
                engine._configure_classes(["cup"])
            collect.assert_not_called()
            self.assertIsNotNone(kept())

    def test_no_collection_when_no_predictor_was_dropped(self):
        # A full collection costs tens of ms with torch loaded; the first prompt
        # after a load or mode switch has no predictor to free.
        engine = OpenVocabEngine()
        engine.model = FakeYOLOE()
        engine.model.predictor = None
        with patch.object(OpenVocabEngine, "_release_dropped_predictor") as release:
            engine._configure_classes(["cup"])
        release.assert_not_called()
        self.assertEqual(engine.model.encoded, [["cup"]])
        visual = VisualPromptEngine()
        visual.model = FakeYOLOE()
        visual.model.predictor = None
        visual.visual_embedding = object()
        with patch.object(OpenVocabEngine, "_release_dropped_predictor") as release:
            visual.release_visual_prompt()
        release.assert_not_called()

    def test_without_the_release_the_dropped_predictor_would_survive(self):
        engine = OpenVocabEngine()
        engine.model = FakeYOLOE()
        with AutomaticGcDisabled(), patch.object(OpenVocabEngine, "_release_dropped_predictor"):
            dropped = weakref.ref(engine.model.predictor)
            engine._configure_classes(["cup"])
            self.assertIsNotNone(dropped())
        gc.collect()

    def test_cuda_cache_is_emptied_only_for_a_cuda_engine(self):
        import torch
        engine = OpenVocabEngine()
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "empty_cache") as empty:
            engine.device = "cpu"
            engine._release_dropped_predictor()
            empty.assert_not_called()
            engine.device = "cuda:0"
            engine._release_dropped_predictor()
            empty.assert_called_once()

    def test_visual_prompt_install_and_release_free_the_old_predictor(self):
        engine = VisualPromptEngine()
        engine.imgsz = 64
        engine.model = FakeYOLOE()
        reference = np.zeros((20, 40, 3), np.uint8)
        embedding = MagicMock()
        safety = types.SimpleNamespace(ResourceLimitError=RuntimeError, preflight_inference=MagicMock(),
                                       handle_cuda_oom=MagicMock(return_value=False))
        with AutomaticGcDisabled(), patch.dict("sys.modules", {"resource_safety": safety}), \
                patch.object(engine, "_encode_visual_prompt", return_value=embedding):
            text_predictor = weakref.ref(engine.model.predictor)
            self.assertEqual(engine.set_visual_prompt(reference, (5, 5, 15, 15), "cup"), "SUCCESS")
            self.assertIsNone(text_predictor())
            engine.model.predictor = CyclicPredictor()
            visual_predictor = weakref.ref(engine.model.predictor)
            engine.release_visual_prompt()
            self.assertIsNone(visual_predictor())
        self.assertEqual(engine.model.model.names, {})

    def test_switching_text_to_visual_frees_the_text_predictor(self):
        window = AIApp()
        self.addCleanup(window.deleteLater)
        window.engine.model = model = FakeYOLOE()
        with AutomaticGcDisabled():
            text_predictor = weakref.ref(model.predictor)
            window._switch_prompt_engine(True)
            self.assertIsNone(text_predictor())
        self.assertIsInstance(window.engine, VisualPromptEngine)
        self.assertIs(window.engine.model, model)
        window.engine.model = None


if __name__ == "__main__":
    unittest.main()
