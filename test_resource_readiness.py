"""Standard-library-only tests for fail-closed resource waiting and continuity."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import uuid

from tools.check_resource_readiness import (
    POLICY, blocking_processes, check_and_record, collect_observation,
    executed_script, next_wait_state, query_nvidia, write_json_atomic,
)


BOOT = str(uuid.uuid4())


def fake_proc(root, uptime=2000, ram=8000, load=2):
    boot = root / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text(BOOT)
    (root / "uptime").write_text(f"{uptime} 0\n")
    (root / "meminfo").write_text(f"MemAvailable: {ram * 1024} kB\n")
    (root / "loadavg").write_text(f"{load} 0 0 1/100 1\n")


def runner_with(gpu="0, GPU-test, 7000, 50\n", processes="", returncode=0):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 5
        return SimpleNamespace(returncode=returncode, stdout=processes if "compute-apps" in command[1] else gpu)
    return run


def clear_observation(uptime=2000, boot=BOOT):
    return {"boot_id": boot, "uptime_seconds": uptime, "eligible": True, "checks": {"all": True}, "errors": []}


class ProcessTests(unittest.TestCase):
    def test_script_position_ignores_rg_monitor_and_shell_command_patterns(self):
        self.assertEqual(executed_script(["python", "-u", "/jobs/run_blast3r_online_with_priors.py"]), "/jobs/run_blast3r_online_with_priors.py")
        self.assertEqual(executed_script(["bash", "/jobs/n02_build_maps.sh"]), "/jobs/n02_build_maps.sh")
        for argv in (["rg", "run_blast3r_online_with_priors.py"], ["python", "-c", "print('main.py')"],
                     ["bash", "-lc", "rg main.py"], ["python", "tools/check_resource_readiness.py"]):
            self.assertNotIn(Path(executed_script(argv) or "").name, {"main.py", "run_blast3r_online_with_priors.py", "n02_build_maps.sh"})

    def test_real_cmdline_scan_separates_other_projects_and_light_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            unrelated = root / "other"
            project.mkdir()
            unrelated.mkdir()
            cases = {101: (["python", "/jobs/run_blast3r_online_with_priors.py"], unrelated),
                     102: (["rg", "main.py"], project), 103: (["python", "main.py"], project),
                     104: (["python", "main.py"], unrelated), 105: (["python", "tools/prepare_training_probe.py"], project),
                     106: (["python", "tools/train_p2_probe.py"], project),
                     107: (["python", "tools/check_resource_readiness.py"], project)}
            cwd_fallbacks = {}
            for pid, (argv, cwd) in cases.items():
                path = root / str(pid)
                path.mkdir()
                (path / "cmdline").write_bytes(b"\0".join(token.encode() for token in argv) + b"\0")
                try:
                    (path / "cwd").symlink_to(cwd, target_is_directory=True)
                except OSError as exc:
                    if getattr(exc, "winerror", None) != 1314:
                        raise
                    # Windows without Developer Mode cannot create the fake /proc
                    # cwd links. Keep real cmdline files and mock only their targets.
                    cwd_fallbacks[path / "cwd"] = cwd
            native_resolve = Path.resolve

            def fixture_resolve(path, *args, **kwargs):
                return native_resolve(cwd_fallbacks.get(path, path), *args, **kwargs)

            with patch.object(Path, "resolve", new=fixture_resolve):
                blockers, errors = blocking_processes(root, project, own_pid=999)
            self.assertEqual(errors, [])
            self.assertEqual({item["pid"] for item in blockers}, {101, 103, 106})


class ObservationTests(unittest.TestCase):
    def test_thresholds_clear_and_reboot_cooldown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_proc(root)
            observation = collect_observation(root, runner=runner_with(), cpu_count=8)
            self.assertTrue(observation["eligible"])
            self.assertEqual(observation["errors"], [])
            (root / "uptime").write_text("200 0\n")
            observation = collect_observation(root, runner=runner_with(), cpu_count=8)
            self.assertFalse(observation["eligible"])
            self.assertFalse(observation["checks"]["boot_cooldown"])

    def test_compute_process_low_memory_high_temp_ram_and_load_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_proc(root, ram=5000, load=8)
            observation = collect_observation(root, runner=runner_with(gpu="0, GPU-test, 5000, 71\n", processes="77, python, GPU-test\n"), cpu_count=8)
            self.assertFalse(observation["eligible"])
            for name in ("ram_available", "cpu_load", "gpu_memory", "gpu_temperature", "no_compute_processes"):
                self.assertFalse(observation["checks"][name])

    def test_missing_or_unsupported_queries_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_proc(root)
            for runner in (runner_with(returncode=1), runner_with(gpu="0, GPU-test, N/A, 50\n"), runner_with(gpu="")):
                observation = collect_observation(root, runner=runner, cpu_count=8)
                self.assertFalse(observation["eligible"])
                self.assertTrue(observation["errors"])
            (root / "meminfo").unlink()
            self.assertFalse(collect_observation(root, runner=runner_with(), cpu_count=8)["eligible"])


class ContinuityTests(unittest.TestCase):
    def test_ten_minutes_same_boot_required(self):
        first = next_wait_state(None, clear_observation(2000))
        self.assertFalse(first["ready"])
        second = next_wait_state(first, clear_observation(2300))
        self.assertFalse(second["ready"])
        third = next_wait_state(second, clear_observation(2600))
        self.assertTrue(third["ready"])
        self.assertEqual(third["continuous_clear_seconds"], 600)

    def test_reboot_failure_long_gap_and_backward_uptime_reset(self):
        first = next_wait_state(None, clear_observation(2000))
        for observation in (clear_observation(2700), clear_observation(2600, str(uuid.uuid4())), clear_observation(1900)):
            state = next_wait_state(first, observation)
            self.assertFalse(state["ready"])
            self.assertEqual(state["continuous_clear_seconds"], 0)
        failed = clear_observation(2300)
        failed.update(eligible=False, checks={"gpu_memory": False})
        state = next_wait_state(first, failed)
        self.assertIsNone(state["clear_since_uptime_seconds"])
        self.assertFalse(next_wait_state(state, clear_observation(2600))["ready"])

    def test_atomic_record_and_corrupt_state_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            proc.mkdir()
            fake_proc(proc)
            state_path = root / "state.json"
            state = check_and_record(state_path, proc_root=proc, runner=runner_with(), cpu_count=8)
            self.assertFalse(state["ready"])
            self.assertEqual(json.loads(state_path.read_text())["clear_since_uptime_seconds"], 2000)
            self.assertTrue((root / "latest_observation.json").is_file())
            self.assertFalse(list(root.glob("*.part")))
            state_path.write_text("corrupted")
            state = check_and_record(state_path, proc_root=proc, runner=runner_with(), cpu_count=8)
            self.assertFalse(state["ready"])
            self.assertIsNone(state["clear_since_uptime_seconds"])
            self.assertTrue(state["observation"]["errors"])


if __name__ == "__main__":
    unittest.main()
