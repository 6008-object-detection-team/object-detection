"""Resource guards verified with mocked CUDA and resource-only subprocesses."""
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import resource_safety as safety


class ResourceSafetyTests(unittest.TestCase):
    def test_windows_peak_rss_uses_peak_working_set_in_mib(self):
        fake = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(
            memory_info=lambda: types.SimpleNamespace(peak_wset=512 * safety.MIB)))
        with patch.object(safety.os, "name", "nt"), patch.dict(sys.modules, {"psutil": fake}):
            self.assertEqual(safety.process_max_rss_mib(), 512)

    def fake_torch(self, free_mb=7000, total_mb=8192, available=True):
        cuda = MagicMock()
        cuda.is_available.return_value = available
        cuda.mem_get_info.return_value = (free_mb * safety.MIB, total_mb * safety.MIB)
        return types.SimpleNamespace(cuda=cuda)

    def test_low_ram_stops_before_gpu_lease_and_torch_import(self):
        fake = self.fake_torch()
        with patch.object(safety, "available_ram_mb", return_value=safety.MIN_AVAILABLE_RAM_MB - 1), \
                patch.object(safety, "_acquire_gpu_lease") as acquire, \
                patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaises(safety.ResourceLimitError):
                safety.prepare_device("CUDA")
        acquire.assert_not_called()
        fake.cuda.is_available.assert_not_called()
        fake.cuda.mem_get_info.assert_not_called()

    def test_cpu_prepare_preflight_and_lease_release_need_no_torch_import(self):
        # A fresh interpreter catches implicit imports which a populated test
        # process could conceal. Only the tiny safety module is imported.
        script = """
import sys
sys.path.insert(0, sys.argv[1])
import resource_safety as safety
safety.available_ram_mb = lambda: 4096
assert 'torch' not in sys.modules
assert safety.prepare_device('CPU') == 'cpu'
safety.preflight_inference('cpu', 640, (480, 640, 3))
safety.release_gpu_lease()
assert 'torch' not in sys.modules
print('CPU_NO_TORCH')
"""
        result = subprocess.run([sys.executable, "-c", script, str(Path(safety.__file__).parent)],
                                check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.stdout.strip(), "CPU_NO_TORCH")

    def test_cuda_allocation_budget_is_exactly_three_gib(self):
        fake = self.fake_torch()
        with patch.object(safety, "available_ram_mb", return_value=4096), \
                patch.object(safety, "_acquire_gpu_lease") as acquire, \
                patch.dict(sys.modules, {"torch": fake}):
            self.assertEqual(safety.prepare_device("CUDA"), "cuda:0")
        acquire.assert_called_once()
        fraction, device = fake.cuda.set_per_process_memory_fraction.call_args.args
        self.assertEqual(device, 0)
        self.assertEqual(fraction * 8192, 3072)

    def test_low_free_vram_releases_lease_without_setting_budget(self):
        fake = self.fake_torch(free_mb=4095)
        with patch.object(safety, "available_ram_mb", return_value=4096), \
                patch.object(safety, "_acquire_gpu_lease"), \
                patch.object(safety, "release_gpu_lease") as release, \
                patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaises(safety.ResourceLimitError):
                safety.prepare_device("cuda:0")
        release.assert_called_once()
        fake.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_unavailable_cuda_releases_lease(self):
        fake = self.fake_torch(available=False)
        with patch.object(safety, "available_ram_mb", return_value=4096), \
                patch.object(safety, "_acquire_gpu_lease"), \
                patch.object(safety, "release_gpu_lease") as release, \
                patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaisesRegex(safety.ResourceLimitError, "CUDA_UNAVAILABLE"):
                safety.prepare_device("cuda")
        release.assert_called_once()
        fake.cuda.mem_get_info.assert_not_called()

    def test_invalid_or_oversized_inputs_stop_before_gpu_checks(self):
        fake = self.fake_torch()
        with patch.object(safety, "_check_ram") as ram, patch.dict(sys.modules, {"torch": fake}):
            for value in [0, 31, 961, 1280, True, 640.5, "invalid", float("nan")]:
                with self.subTest(imgsz=value), self.assertRaises(safety.ResourceLimitError):
                    safety.preflight_inference("cuda:0", value, (480, 640, 3))
            with self.assertRaises(safety.ResourceLimitError):
                safety.preflight_inference("cuda:0", 640, (4001, 4000, 3))
        ram.assert_not_called()
        fake.cuda.mem_get_info.assert_not_called()

    def test_source_limit_boundary_and_free_vram_inference_guard(self):
        fake = self.fake_torch(free_mb=safety.MIN_FREE_GPU_MB - 1)
        with patch.object(safety, "available_ram_mb", return_value=4096), patch.dict(sys.modules, {"torch": fake}):
            safety.preflight_inference("cpu", 960, (4000, 4000, 3))
            fake.cuda.mem_get_info.assert_not_called()
            with self.assertRaises(safety.ResourceLimitError):
                safety.preflight_inference("cuda:0", 640, (480, 640, 3))

    def test_gpu_lease_blocks_other_process_and_release_allows_it(self):
        script = """
import sys
sys.path.insert(0, sys.argv[1])
import resource_safety as safety
safety.__file__ = sys.argv[2]
try:
    safety._acquire_gpu_lease()
except safety.ResourceLimitError:
    print('BLOCKED')
else:
    safety.release_gpu_lease()
    print('ACQUIRED')
assert 'torch' not in sys.modules
"""
        original_file = safety.__file__
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(safety, "__file__", str(Path(temp) / "resource_safety.py")), \
                patch.object(safety, "_gpu_lease", None):
            command = [sys.executable, "-c", script, str(Path(original_file).parent), safety.__file__]
            try:
                safety._acquire_gpu_lease()
                blocked = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
                self.assertEqual(blocked.stdout.strip(), "BLOCKED")
                safety.release_gpu_lease()
                allowed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
                self.assertEqual(allowed.stdout.strip(), "ACQUIRED")
            finally:
                safety.release_gpu_lease()

    def test_cuda_oom_stops_job_and_clears_unused_cache_without_retry(self):
        fake = self.fake_torch()
        failure = RuntimeError("CUDA out of memory. Tried to allocate test bytes")
        infer = MagicMock(side_effect=failure)
        guarded = safety.guarded_inference(infer)
        with patch.object(safety.gc, "collect"), patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaises(safety.ResourceLimitError) as caught:
                guarded("frame")
        self.assertIs(caught.exception.__cause__, failure)
        infer.assert_called_once_with("frame")
        fake.cuda.empty_cache.assert_called_once()

    def test_other_errors_are_preserved_without_gpu_cleanup(self):
        fake = self.fake_torch()
        failure = ValueError("invalid prompt")
        guarded = safety.guarded_inference(MagicMock(side_effect=failure))
        with patch.dict(sys.modules, {"torch": fake}):
            with self.assertRaises(ValueError) as caught:
                guarded()
        self.assertIs(caught.exception, failure)
        fake.cuda.empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
