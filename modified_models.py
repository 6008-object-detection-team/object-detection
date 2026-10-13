"""One committed modified model per family, with bounded staging and rollback.

Only validated candidates replace the fixed checkpoint. Audit JSON is retained,
not successive completed model copies. Filesystem locks are independent of the
normal GPU lease; temporary checkpoint backups exist only during publication.
"""
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from feedback import PROJECT, atomic_json, safe_path, sha256_file
from model_families import MODEL_FAMILIES, model_family
from i18n import message


class ModifiedModels:
    def __init__(self, store_root):
        self.store_root = Path(store_root).absolute()
        self.root = safe_path(self.store_root, "models")

    def slot(self, family):
        if family not in MODEL_FAMILIES:
            raise ValueError("请选择支持纠错的 YOLOE 基础型号")
        return safe_path(self.root, family)

    def current(self, family):
        slot = self.slot(family)
        if safe_path(self.root, family + "/publish.pending.json").exists():
            raise ValueError("修改版正在提交或上次提交中断；不加载不完整权重，请等待或重新执行更新以恢复。")
        metadata = safe_path(self.root, family + "/current.json")
        if not metadata.is_file():
            return None
        row = json.loads(metadata.read_text(encoding="utf-8"))
        checkpoint = safe_path(self.root, family + "/modified.pt")
        if row.get("model_family") != family or not self.ready(row) or not checkpoint.is_file() \
                or sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            raise ValueError("修改版校验失败；不回退到未经修改的模型，请重新检查或更新。")
        return row

    @staticmethod
    def ready(row):
        return row.get("status") == "ready" and all(row.get(key) for key in
            ("text_prompt", "visual_prompt", "parent_unchanged", "frozen_modules_unchanged", "detection_heads_changed"))

    def listed(self):
        result = []
        for family, title in MODEL_FAMILIES.items():
            row = self.current(family)
            if row:
                result.append((title + "修改版", str(self.slot(family)/"modified.pt"), row))
        # Read-only compatibility: old training runs are not deleted or rewritten.
        # Only their latest eligible run is exposed if no fixed slot exists yet.
        existing = {row[2]["model_family"] for row in result}
        legacy = {}
        if self.root.exists():
            for path in sorted(self.root.glob("*/version.json")):
                safe_path(self.root, path.relative_to(self.root))
                row = json.loads(path.read_text(encoding="utf-8"))
                family = row.get("model_family") or model_family(row.get("base_model", row.get("parent", "")))
                if family not in MODEL_FAMILIES or family in existing or not self.ready(row):
                    continue
                checkpoint = safe_path(self.root, path.parent.relative_to(self.root)/"corrected.pt")
                if checkpoint.is_file() and sha256_file(checkpoint) == row.get("checkpoint_sha256"):
                    legacy[family] = (MODEL_FAMILIES[family] + "修改版", str(checkpoint), dict(row, model_family=family))
        return result + list(legacy.values())

    @contextmanager
    def updating(self, family):
        self.slot(family)
        path = safe_path(self.root, ".locks/" + family + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    handle.write(b"0"); handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            self.recover(family)
            yield
        except (BlockingIOError, PermissionError) as exc:
            if not acquired:
                raise RuntimeError("这个基础模型已有更新任务；不会并发修改同一份权重。") from exc
            raise
        finally:
            if acquired:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def recover(self, family):
        """Recover only while the caller owns the writer lock, never on listing."""
        slot = self.slot(family)
        journal = slot/"publish.pending.json"
        if not journal.exists():
            return
        row = json.loads(journal.read_text(encoding="utf-8"))
        backup = safe_path(self.root, row["backup"])
        if not backup.is_relative_to(self.root/".staging") or backup.name != "prior.pt" \
                or row["new"].get("model_family") != family or not self.ready(row["new"]):
            raise ValueError("修改版提交记录损坏，不执行恢复或清理")
        checkpoint = slot/"modified.pt"
        new, previous = row["new"], row["previous"]
        if checkpoint.exists() and sha256_file(checkpoint) == new["checkpoint_sha256"]:
            atomic_json(slot/"current.json", new)
        elif previous:
            if not backup.is_file() or sha256_file(backup) != previous["checkpoint_sha256"]:
                raise ValueError("修改版更新恢复失败：上次可用权重备份不完整，不自动覆盖。")
            os.replace(backup, checkpoint)
            atomic_json(slot/"current.json", previous)
        else:
            raise ValueError("修改版首次提交中断且候选不完整；请检查本地训练记录。")
        journal.unlink()
        backup.unlink(missing_ok=True)

    def parent_for(self, family):
        # Reuse the one latest modified model and replay all reviewed samples.
        current = self.current(family)
        if current:
            return self.slot(family)/"modified.pt", current
        legacy = next((row for _, _, row in self.listed() if row["model_family"] == family), None)
        if legacy:
            path = next(path for _, path, row in self.listed() if row["model_family"] == family)
            return Path(path), legacy
        return PROJECT/(family + "-seg.pt"), None

    def publish(self, family, job, status, previous=None):
        """Caller owns updating(). Commit validated bytes, with crash journal."""
        job = safe_path(self.root, Path(job).absolute().relative_to(self.root))
        if not job.is_relative_to(self.root/".staging") or not self.ready(status):
            raise ValueError("只提交本项目临时目录中通过完整检查的修改候选")
        candidate = safe_path(self.root, job.relative_to(self.root)/"corrected.pt")
        if sha256_file(candidate) != status.get("checkpoint_sha256"):
            raise ValueError("训练候选被修改，不替换已有修改版")
        slot = self.slot(family)
        slot.mkdir(parents=True, exist_ok=True)
        current = self.current(family)
        if current and current["checkpoint_sha256"] != (previous or {}).get("checkpoint_sha256"):
            raise ValueError("修改版在训练期间发生变化，不覆盖新结果")
        row = dict(status, schema=2, model_family=family, display_name=MODEL_FAMILIES[family] + "修改版",
                   demo_exposure=bool(status.get("demo_exposure") or (previous or {}).get("demo_exposure")),
                   base_model=str(PROJECT/(family + "-seg.pt")),
                   update_count=int((previous or {}).get("update_count", 0)) + 1,
                   updated_at=datetime.now(timezone.utc).isoformat(),
                   run_id=job.name, checkpoint=str(slot/"modified.pt"))
        backup = job/"prior.pt"
        if current:
            shutil.copyfile(slot/"modified.pt", backup)
            if sha256_file(backup) != current["checkpoint_sha256"]:
                raise ValueError("旧修改版备份校验失败，不提交新权重")
        incoming = safe_path(self.root, family + "/.incoming-" + uuid.uuid4().hex + ".pt")
        try:
            shutil.copyfile(candidate, incoming)
            if sha256_file(incoming) != row["checkpoint_sha256"]:
                raise ValueError("修改版复制校验失败")
            atomic_json(slot/"publish.pending.json", {"new": row, "previous": current,
                         "backup": str(backup.relative_to(self.root))})
            os.replace(incoming, slot/"modified.pt")
            atomic_json(slot/"current.json", row)
            (slot/"publish.pending.json").unlink()
            backup.unlink(missing_ok=True)
            return row
        except BaseException:
            # Restore the last committed model on ordinary failures. On process
            # death the journal lets the next locked update recover consistently.
            if (slot/"publish.pending.json").exists() and current and backup.exists():
                os.replace(backup, slot/"modified.pt")
                if json.loads((slot/"current.json").read_text(encoding="utf-8")) != current:
                    atomic_json(slot/"current.json", current)
                (slot/"publish.pending.json").unlink()
            elif (slot/"publish.pending.json").exists() and current is None:
                # These paths did not exist before this first publication;
                # remove only our failed first commit, not an existing model.
                if (slot/"modified.pt").exists() and sha256_file(slot/"modified.pt") == row["checkpoint_sha256"]:
                    (slot/"modified.pt").unlink()
                (slot/"current.json").unlink(missing_ok=True)
                (slot/"publish.pending.json").unlink()
            raise
        finally:
            incoming.unlink(missing_ok=True)

    def cleanup_job_weights(self, job):
        job = safe_path(self.root, Path(job).absolute().relative_to(self.root))
        if not job.is_relative_to(self.root/".staging"):
            raise ValueError("不清理临时训练目录以外的模型文件")
        for name in ("parent-copy.pt", "corrected.pt"):
            safe_path(self.root, job.relative_to(self.root)/name).unlink(missing_ok=True)
