"""Live prompt edits reach the running detection once per typing pause, or at
once on Enter. Offscreen Qt, fake worker and model, no GPU work."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from engines import OpenVocabEngine, PromptFreeEngine
from main import AIApp
from test_predictor_release import FakeYOLOE


class RecordingWorker:
    """Records each class list the GUI hands to the running worker."""

    def __init__(self, class_names):
        self._class_names = class_names
        self.assigned = []

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, value):
        self.assigned.append(list(value))
        self._class_names = value


class DebouncedPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = AIApp()
        self.addCleanup(self.window.deleteLater)
        self.window.prompt_input.setText("person, ")
        self.window.prompt_apply_timer.stop()
        self.worker = RecordingWorker(["person"])
        self.window.worker = self.worker
        self.window.current_video_label = self.window.video_label_tab1
        self.window.active_detection_request = {"target_label": self.window.video_label_tab1,
                                                "prompts": ["person"]}
        self.addCleanup(setattr, self.window, "worker", None)

    def wait_past_debounce(self):
        QTest.qWait(self.window.prompt_apply_timer.interval() + 250)

    def test_typing_burst_reaches_the_worker_once_after_a_pause(self):
        timer = self.window.prompt_apply_timer
        self.assertTrue(timer.isSingleShot())
        self.assertGreaterEqual(timer.interval(), 300)
        QTest.keyClicks(self.window.prompt_input, "cell phone")
        self.assertEqual(self.worker.assigned, [])  # half-typed prompts never reach the model
        self.assertTrue(timer.isActive())
        self.wait_past_debounce()
        self.assertEqual(self.worker.assigned, [["person", "cell phone"]])
        self.assertEqual(self.window.active_detection_request["prompts"], ["person", "cell phone"])
        self.assertFalse(timer.isActive())

    def test_enter_applies_at_once_without_a_second_update(self):
        QTest.keyClicks(self.window.prompt_input, "cup")
        QTest.keyClick(self.window.prompt_input, Qt.Key_Return)
        self.assertEqual(self.worker.assigned, [["person", "cup"]])
        self.assertFalse(self.window.prompt_apply_timer.isActive())
        self.wait_past_debounce()
        self.assertEqual(self.worker.assigned, [["person", "cup"]])

    def test_running_engine_reencodes_once_per_typing_burst(self):
        engine = OpenVocabEngine()
        engine.model = FakeYOLOE()
        engine._configure_classes(["person"])
        self.window.engine = engine
        for character in "cell phone":
            QTest.keyClick(self.window.prompt_input, character)
            engine._configure_classes(self.worker.class_names)  # one live frame per key
        self.wait_past_debounce()
        engine._configure_classes(self.worker.class_names)
        self.assertEqual(engine.model.encoded, [["person"], ["person", "cell phone"]])

    def test_prompt_free_guard_stops_once_after_the_pause(self):
        self.window.engine = PromptFreeEngine()
        self.worker.class_names = []
        self.worker.assigned.clear()
        self.window.prompt_input.clear()
        self.window.prompt_apply_timer.stop()
        with patch.object(self.window, "stop_video_stream", return_value=True) as stop:
            QTest.keyClicks(self.window.prompt_input, "cup")
            stop.assert_not_called()
            self.wait_past_debounce()
        stop.assert_called_once()
        self.assertEqual(self.worker.assigned, [])

    def test_next_image_within_the_pause_runs_with_the_new_prompt(self):
        window = self.window
        window.worker = None  # the previous image finished; the request is kept for navigation
        window.image_paths, window.image_index = ["a.jpg", "b.jpg"], 0
        window.active_detection_request.update(template=None)
        with patch.object(window, "_start_stream") as start, patch.object(window, "_update_image_position"):
            window.prompt_input.setText("cup")
            self.assertTrue(window.prompt_apply_timer.isActive())
            window.next_image()  # within 400 ms of the last key
            self.assertFalse(window.prompt_apply_timer.isActive())
            self.wait_past_debounce()
        start.assert_called_once()
        self.assertEqual(start.call_args.args[1], ["cup"])
        self.assertEqual(window.active_detection_request["prompts"], ["cup"])

    def test_explicit_text_argument_still_applies_immediately(self):
        QTest.keyClicks(self.window.prompt_input, "bag")
        self.window.update_worker_classes("bottle")
        self.assertEqual(self.worker.assigned, [["bottle"]])
        self.assertFalse(self.window.prompt_apply_timer.isActive())


if __name__ == "__main__":
    unittest.main()
