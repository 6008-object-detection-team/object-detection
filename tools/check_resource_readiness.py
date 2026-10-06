#!/usr/bin/env python3
"""Read-only resource wait check; never launches experiments or calendar reminders.

This records whether the same boot has met all safety gates for ten minutes.
Absence of a BLAST3R process does not establish successful job completion.
Only standard-library modules are imported; no Torch, CUDA initialization,
process termination, inference, training, or scheduling occurs here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE = PROJECT_ROOT / "work/additional_experiments/resource_wait/state.json"
POLICY = {"gpu_index": 0, "minimum_gpu_free_mib": 6144, "minimum_ram_available_mib": 6144,
          "maximum_gpu_temperature_c": 70, "minimum_boot_uptime_seconds": 900,
          "maximum_load_per_cpu": 0.5, "continuous_clear_seconds": 600,
          "maximum_sample_gap_seconds": 660}
PROTECTED_SCRIPTS = {"run_blast3r_online_with_priors.py", "n02_build_maps.sh"}
PROJECT_MODEL_SCRIPTS = {"main.py", "evaluate_mini_dataset.py", "benchmark_detection.py",
                         "record_demo.py", "validate_improvements.py", "run_additional_experiments.py",
                         "train_p2_probe.py", "run_app.sh", "run_mini_evaluation.sh", "run_experiments.sh"}


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def executed_script(argv):
    """Only examine executable/script positions, never grep pattern arguments."""
    if not argv:
        return None
    executable = Path(argv[0]).name
    if executable in PROTECTED_SCRIPTS | PROJECT_MODEL_SCRIPTS:
        return argv[0]
    if executable.startswith("python"):
        for token in argv[1:]:
            if token in ("-c", "-m"):
                return None
            if token.startswith("-"):
                continue
            return token
    if executable in {"bash", "sh", "dash", "zsh"}:
        for token in argv[1:]:
            if token in {"-c", "-lc", "-ic", "-l" + "c"}:
                return None
            if token.startswith("-"):
                continue
            return token
    return None


def blocking_processes(proc_root, project_root, own_pid=None):
    blockers, errors = [], []
    own_pid = os.getpid() if own_pid is None else own_pid
    try:
        candidates = list(proc_root.iterdir())
    except OSError as exc:
        return [], [f"process_scan_failed:{type(exc).__name__}"]
    project_root = project_root.resolve()
    for directory in candidates:
        if not directory.name.isdigit() or int(directory.name) == own_pid:
            continue
        try:
            payload = (directory / "cmdline").read_bytes()
            argv = [token.decode("utf-8", errors="replace") for token in payload.split(b"\0") if token]
            entrypoint = executed_script(argv)
            if entrypoint is None:
                continue
            name = Path(entrypoint).name
            if name in PROTECTED_SCRIPTS:
                blockers.append({"pid": int(directory.name), "script": name, "kind": "protected_workload"})
                continue
            if name not in PROJECT_MODEL_SCRIPTS:
                continue
            try:
                cwd = (directory / "cwd").resolve(strict=True)
            except (FileNotFoundError, ProcessLookupError):
                # A process can exit between reading argv and its cwd.
                continue
            script_path = Path(entrypoint) if Path(entrypoint).is_absolute() else cwd / entrypoint
            script_path = script_path.resolve()
            if script_path.is_relative_to(project_root) or cwd.is_relative_to(project_root):
                blockers.append({"pid": int(directory.name), "script": name, "kind": "project_model_workload"})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            errors.append(f"process_scan_pid_{directory.name}:{type(exc).__name__}")
    return sorted(blockers, key=lambda item: item["pid"]), errors


def query_nvidia(runner=None):
    runner = subprocess.run if runner is None else runner
    commands = (
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,temperature.gpu", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader,nounits"],
    )
    outputs = []
    for command in commands:
        completed = runner(command, capture_output=True, text=True, timeout=5, check=False)
        if completed.returncode != 0:
            raise RuntimeError("nvidia-smi query failed")
        outputs.append(completed.stdout)
    gpus = []
    for row in csv.reader(io.StringIO(outputs[0])):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError("Unexpected GPU query columns")
        index, gpu_uuid, memory, temperature = map(str.strip, row)
        gpu = {"index": int(index), "uuid": gpu_uuid, "free_mib": int(memory), "temperature_c": int(temperature)}
        if gpu["index"] < 0 or gpu["free_mib"] < 0 or not 0 <= gpu["temperature_c"] <= 150 or not gpu_uuid:
            raise ValueError("Invalid GPU observation")
        gpus.append(gpu)
    if len({gpu["index"] for gpu in gpus}) != len(gpus):
        raise ValueError("Duplicate GPU indices")
    processes = []
    for row in csv.reader(io.StringIO(outputs[1])):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("Unexpected compute query columns")
        pid, name, gpu_uuid = map(str.strip, row)
        if int(pid) <= 0 or not name or not gpu_uuid:
            raise ValueError("Invalid GPU compute-process observation")
        processes.append({"pid": int(pid), "process": Path(name).name, "gpu_uuid": gpu_uuid})
    return gpus, processes


def collect_observation(proc_root=Path("/proc"), project_root=PROJECT_ROOT, runner=None, cpu_count=None):
    observation = {"observed_utc": datetime.now(timezone.utc).isoformat(), "errors": [],
                   "boot_id": None, "uptime_seconds": None, "ram_available_mib": None,
                   "cpu_count": None, "load_one_minute": None, "load_per_cpu": None,
                   "gpu": None, "gpu_compute_processes": [], "blocking_processes": [],
                   "job_completion": "unknown; process absence is not completion evidence"}
    try:
        boot_path = proc_root / "sys/kernel/random/boot_id"
        boot_before = str(uuid.UUID(boot_path.read_text().strip()))
        uptime = float((proc_root / "uptime").read_text().split()[0])
        if not math.isfinite(uptime) or uptime < 0:
            raise ValueError("Invalid uptime")
        observation.update(boot_id=boot_before, uptime_seconds=uptime)
        memory = {line.split(":", 1)[0]: line.split(":", 1)[1].split()
                  for line in (proc_root / "meminfo").read_text().splitlines() if ":" in line}
        if memory["MemAvailable"][1] != "kB":
            raise ValueError("Unknown MemAvailable unit")
        available = int(memory["MemAvailable"][0]) / 1024
        if available < 0:
            raise ValueError("Negative available RAM")
        observation["ram_available_mib"] = available
        load = float((proc_root / "loadavg").read_text().split()[0])
        count = os.cpu_count() if cpu_count is None else cpu_count
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or not math.isfinite(load) or load < 0:
            raise ValueError("Invalid CPU/load observation")
        observation.update(cpu_count=count, load_one_minute=load, load_per_cpu=load / count)
        if str(uuid.UUID(boot_path.read_text().strip())) != boot_before:
            raise RuntimeError("Boot changed during observation")
    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        observation["errors"].append(f"host_query_failed:{type(exc).__name__}")
    blockers, errors = blocking_processes(proc_root, project_root)
    observation["blocking_processes"] = blockers
    observation["errors"].extend(errors)
    try:
        gpus, compute = query_nvidia(runner)
        observation["gpu_compute_processes"] = compute
        observation["gpu"] = next(gpu for gpu in gpus if gpu["index"] == POLICY["gpu_index"])
    except (OSError, ValueError, RuntimeError, StopIteration, subprocess.TimeoutExpired) as exc:
        observation["errors"].append(f"gpu_query_failed:{type(exc).__name__}")
    gpu = observation["gpu"] or {}
    observation["checks"] = {
        "queries_succeeded": not observation["errors"],
        "boot_cooldown": observation["uptime_seconds"] is not None and observation["uptime_seconds"] >= POLICY["minimum_boot_uptime_seconds"],
        "ram_available": observation["ram_available_mib"] is not None and observation["ram_available_mib"] >= POLICY["minimum_ram_available_mib"],
        "cpu_load": observation["load_per_cpu"] is not None and observation["load_per_cpu"] <= POLICY["maximum_load_per_cpu"],
        "gpu_memory": gpu.get("free_mib", -1) >= POLICY["minimum_gpu_free_mib"],
        "gpu_temperature": gpu.get("temperature_c", 1000) <= POLICY["maximum_gpu_temperature_c"],
        "no_compute_processes": not observation["gpu_compute_processes"],
        "no_blocking_workloads": not blockers,
    }
    observation["eligible"] = all(observation["checks"].values())
    return observation


def next_wait_state(previous, observation):
    """Use boot uptime, not adjustable wall time, for the continuous-clear window."""
    state = {"schema_version": 1, "policy": POLICY, "boot_id": observation["boot_id"],
             "last_sample_uptime_seconds": observation["uptime_seconds"], "clear_since_uptime_seconds": None,
             "continuous_clear_seconds": 0.0, "ready": False, "observation": observation,
             "reasons": [name for name, passed in observation["checks"].items() if not passed]}
    if not observation["eligible"]:
        return state
    current = observation["uptime_seconds"]
    since = current
    if previous and previous.get("schema_version") == 1 and previous.get("policy") == POLICY and previous.get("boot_id") == observation["boot_id"]:
        last = previous.get("last_sample_uptime_seconds")
        old_since = previous.get("clear_since_uptime_seconds")
        if (isinstance(last, (int, float)) and isinstance(old_since, (int, float))
                and math.isfinite(last) and math.isfinite(old_since)
                and 0 <= current - last <= POLICY["maximum_sample_gap_seconds"]
                and 0 <= old_since <= last):
            since = old_since
    state["clear_since_uptime_seconds"] = since
    state["continuous_clear_seconds"] = current - since
    state["ready"] = state["continuous_clear_seconds"] >= POLICY["continuous_clear_seconds"]
    if not state["ready"]:
        state["reasons"].append("continuous_clear_window_not_met")
    return state


def check_and_record(state_path=DEFAULT_STATE, observation_path=None, *, proc_root=Path("/proc"), runner=None, cpu_count=None):
    observation_path = observation_path or state_path.with_name("latest_observation.json")
    observation = collect_observation(proc_root=proc_root, runner=runner, cpu_count=cpu_count)
    previous = None
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text())
            if not isinstance(previous, dict):
                raise ValueError("State is not an object")
        except (OSError, ValueError) as exc:
            observation["errors"].append(f"state_read_failed:{type(exc).__name__}")
            observation["checks"]["queries_succeeded"] = False
            observation["eligible"] = False
    state = next_wait_state(previous, observation)
    write_json_atomic(observation_path, observation)
    write_json_atomic(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--observation", type=Path, help="Atomic observation JSON; default next to state")
    args = parser.parse_args()
    try:
        state = check_and_record(args.state, args.observation)
        print(json.dumps({"ready": state["ready"], "reasons": state["reasons"],
                          "continuous_clear_seconds": state["continuous_clear_seconds"],
                          "state_file": str(args.state), "observation": state["observation"]}, ensure_ascii=False, allow_nan=False))
        return 2 if state["observation"]["errors"] else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"ready": False, "errors": [f"check_record_failed:{type(exc).__name__}"],
                          "job_completion": "unknown; no workload is launched"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
