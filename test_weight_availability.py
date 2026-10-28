"""Undownloaded weights: list labels, friendly load refusal and check_environment.

Uses temporary pointer/sparse files and mocked engines; no model is loaded.
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import check_environment
from engines import OpenVocabEngine, PromptFreeEngine
from i18n import render
from tools.fetch_models import weight_state

POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
           b"oid sha256:2d7e2b4b1d2f0c9f1f7f2e1b9a2b9f0c3f6d1b6c1c1e8d7c4b5a6f7e8d9c0b1a\nsize 104857600\n")


def make_weight(root, name, kind):
    path = Path(root) / name
    if kind == "pointer":
        path.write_bytes(POINTER)
    elif kind == "small":
        path.write_bytes(b"\x80\x02" + b"0" * 64)
    elif kind == "ok":
        with path.open("wb") as stream:  # sparse 1 MiB file
            stream.truncate(1024 * 1024)
    return path


class WeightStateTests(unittest.TestCase):
    def test_pointer_missing_incomplete_and_downloaded(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertLess(len(POINTER), 200)
            self.assertEqual(weight_state(make_weight(root, "a.pt", "pointer")), "lfs_pointer")
            self.assertEqual(weight_state(make_weight(root, "b.pt", "small")), "incomplete")
            self.assertEqual(weight_state(make_weight(root, "c.pt", "ok")), "ok")
            self.assertEqual(weight_state(Path(root) / "absent.pt"), "missing")
            os.symlink(Path(root) / "absent.pt", Path(root) / "broken.pt")
            self.assertEqual(weight_state(Path(root) / "broken.pt"), "missing")
            os.mkdir(Path(root) / "folder.pt")
            self.assertEqual(weight_state(Path(root) / "folder.pt"), "missing")


class CheckEnvironmentTests(unittest.TestCase):
    def run_main(self, root, packages=([], [], [])):
        real_check_models, output = check_environment.check_models, io.StringIO()
        with patch.object(check_environment, "check_packages", return_value=packages), \
                patch.object(check_environment, "check_models", side_effect=lambda: real_check_models(root)), \
                redirect_stdout(output):
            code = check_environment.main([])
        return code, output.getvalue()

    def test_optional_pointers_warn_and_exit_zero(self):
        with tempfile.TemporaryDirectory() as root:
            for name in check_environment.REQUIRED_MODELS:
                make_weight(root, name, "ok")
            make_weight(root, "yoloe-11l-seg.pt", "pointer")
            lines, errors, warnings = check_environment.check_models(root)
            self.assertEqual(errors, [])
            self.assertTrue(any("yoloe-11l-seg.pt (optional): Git LFS pointer" in item for item in warnings))
            self.assertTrue(any("yoloe-26x-seg-pf.pt (optional): missing" in item for item in warnings))
            self.assertEqual(len(lines), len(check_environment.REQUIRED_MODELS))
            code, text = self.run_main(root, packages=(["OK numpy: 2"], [], ["sahi (optional, x): ImportError"]))
            self.assertEqual(code, 0)
            self.assertIn("WARN yoloe-11l-seg.pt", text)
            self.assertIn("WARN sahi (optional", text)
            self.assertIn("optional item(s) are missing", text)
            self.assertNotIn("ERROR", text)

    def test_required_pointer_or_package_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            for name in check_environment.REQUIRED_MODELS:
                make_weight(root, name, "ok")
            make_weight(root, "yoloe-26s-seg-pf.pt", "pointer")
            code, text = self.run_main(root)
            self.assertEqual(code, 1)
            self.assertIn("ERROR yoloe-26s-seg-pf.pt (required): Git LFS pointer", text)
            self.assertIn("--models yoloe-26s-seg-pf.pt", text)
            make_weight(root, "yoloe-26s-seg-pf.pt", "ok")
            code, text = self.run_main(root, packages=([], ["torch: No module named 'torch'"], []))
            self.assertEqual(code, 1)
            self.assertIn("ERROR torch", text)

    def test_required_and_optional_sets(self):
        self.assertEqual(set(check_environment.REQUIRED_MODELS),
                         {"yoloe-26s-seg.pt", "yoloe-26s-seg-pf.pt", "mobileclip2_b.ts"})
        self.assertIn("mobileclip_blt.ts", check_environment.OPTIONAL_MODELS)
        self.assertIn("yoloe-11l-seg.pt", check_environment.OPTIONAL_MODELS)
        self.assertFalse(set(check_environment.REQUIRED_MODELS) & set(check_environment.OPTIONAL_MODELS))
        self.assertIn("sahi", check_environment.OPTIONAL_PACKAGES)
        self.assertIn("pycocotools", check_environment.OPTIONAL_PACKAGES)
        self.assertIsInstance(check_environment.package_version("json", "json"), str)


class ModelListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def window(self, missing):
        from main import AIApp
        states = lambda path: "lfs_pointer" if Path(path).name in missing else "ok"
        with patch("main.weight_state", side_effect=states):
            window = AIApp()
        self.addCleanup(window.close)
        return window, states

    def test_undownloaded_entries_are_labelled_and_translated(self):
        window, _ = self.window({"yoloe-11l-seg.pt", "yoloe-26x-seg-pf.pt"})
        index = window.model_combo.findData("yoloe-11l-seg.pt")
        self.assertEqual(window.model_combo.itemText(index), "YOLOE-11L（旧模型对照）（未下载）")
        self.assertNotIn("未下载", window.model_combo.itemText(window.model_combo.findData("yoloe-26s-seg.pt")))
        self.assertTrue(window.model_combo.model().item(index).isEnabled())
        window.ui_language.set_language("en")
        self.assertEqual(window.model_combo.itemText(index), "YOLOE-11L (older baseline) (not downloaded)")
        self.assertIn("(not downloaded)", window.model_combo.itemText(window.model_combo.findData("yoloe-26x-seg-pf.pt")))
        self.assertEqual(window.MODELS["YOLOE-11L (older baseline) (not downloaded)"], "yoloe-11l-seg.pt")

    def test_loading_a_pointer_explains_instead_of_the_unpickling_dialog(self):
        window, states = self.window({"yoloe-11l-seg.pt"})
        previous = window.engine
        with patch("main.weight_state", side_effect=states), \
                patch.object(OpenVocabEngine, "load_model") as load, \
                patch("main.QMessageBox.warning") as warning, patch("main.QMessageBox.critical") as critical:
            self.assertFalse(window._load_model_path("yoloe-11l-seg.pt"))
        load.assert_not_called()
        critical.assert_not_called()
        title, detail = warning.call_args.args[1:3]
        self.assertEqual(render(title, "en"), "Model not downloaded")
        self.assertIn("Git LFS pointer", render(detail, "en"))
        self.assertIn("python tools/fetch_models.py --models yoloe-11l-seg.pt", render(detail, "en"))
        self.assertIn("没有加载或替换任何模型", render(detail, "zh"))
        self.assertIs(window.engine, previous)

    def test_label_is_refreshed_after_weights_arrive_and_load(self):
        window, _ = self.window({"yoloe-26l-seg.pt"})
        index = window.model_combo.findData("yoloe-26l-seg.pt")
        self.assertIn("未下载", window.model_combo.itemText(index))
        with patch("main.weight_state", return_value="ok"), \
                patch.object(OpenVocabEngine, "load_model", return_value="SUCCESS"), \
                patch("torch.cuda.is_available", return_value=False):
            self.assertTrue(window._load_model_path("yoloe-26l-seg.pt"))  # fetched while the app ran
        self.assertNotIn("未下载", window.model_combo.itemText(index))
        self.assertEqual(window.model_combo.currentData(), "yoloe-26l-seg.pt")
        window.ui_language.set_language("en")
        self.assertNotIn("not downloaded", window.model_combo.itemText(index))

    def test_prompt_free_weights_keep_their_own_manifest_check(self):
        window, _ = self.window({"yoloe-26s-seg-pf.pt"})
        with patch("main.weight_state", return_value="lfs_pointer"), \
                patch.object(PromptFreeEngine, "load_model", return_value="SUCCESS") as load, \
                patch("torch.cuda.is_available", return_value=False):
            window._load_model_path("yoloe-26s-seg-pf.pt")
        load.assert_called_once()

    def test_list_rebuild_keeps_the_labels(self):
        window, states = self.window({"yoloe-26x-seg.pt"})
        with patch("main.weight_state", side_effect=states):
            window.refresh_feedback_models()
        self.assertIn("未下载", window.model_combo.itemText(window.model_combo.findData("yoloe-26x-seg.pt")))
        self.assertNotIn("未下载", window.model_combo.itemText(window.model_combo.findData("yoloe-26l-seg.pt")))


if __name__ == "__main__":
    unittest.main()
