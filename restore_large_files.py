"""Restore files split to fit GitHub LFS limits; requires only Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLOCK_SIZE = 8 * 1024 * 1024


def local_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f"Path is outside the project: {relative}")
    return path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BLOCK_SIZE), b""):
            result.update(block)
    return result.hexdigest()


def restore(entry, check_only=False):
    target = local_path(entry["path"])
    if target.exists() and (target.stat().st_size != entry["size"] or digest(target) != entry["sha256"]):
        raise ValueError(f"Existing file differs; it will not be overwritten: {target}")
    temporary = target.with_name(target.name + ".restoring")
    output = None
    try:
        if not check_only and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            output = temporary.open("xb")
        whole_hash = hashlib.sha256()
        whole_size = 0
        for part in entry["parts"]:
            path = local_path(part["path"])
            if not path.is_file() or path.stat().st_size != part["size"]:
                raise ValueError(f"Missing/incomplete part: {path}. Run git lfs pull first.")
            part_hash = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(BLOCK_SIZE), b""):
                    whole_size += len(block)
                    part_hash.update(block)
                    whole_hash.update(block)
                    if output is not None:
                        output.write(block)
            if part_hash.hexdigest() != part["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
        if whole_size != entry["size"] or whole_hash.hexdigest() != entry["sha256"]:
            raise ValueError(f"Reassembled checksum mismatch: {target}")
        if output is not None:
            output.close()
            # Windows rename refuses to replace a file created during restoration.
            temporary.rename(target)
        print(f"OK {entry['path']} ({whole_size:,} bytes, SHA-256 verified)")
    finally:
        if output is not None:
            output.close()
            if temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify every part without restoring files")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "large_files.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        restore(entry, check_only=args.check)


if __name__ == "__main__":
    main()
