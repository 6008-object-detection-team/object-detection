"""Fetch pinned upstream weights, checking size/SHA256 without loading models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(record, folder):
    name = record["file"]
    if Path(name).name != name:
        raise ValueError("Model name must be a basename")
    target = folder / name
    if target.is_file() and target.stat().st_size == record["size"] and sha256(target) == record["sha256"]:
        print(f"Verified: {name}", flush=True)
        return target
    # Keep any existing file until its validated replacement is complete.
    temporary = folder / (name + ".downloading")
    request = urllib.request.Request(record["url"], headers={"User-Agent": "object-detection-lab-model-fetcher"})
    digest, size = hashlib.sha256(), 0
    try:
        print(f"Downloading: {name} ({record['size'] / 1024**2:.1f} MiB)", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            reported = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                size += len(chunk)
                if size > record["size"]:
                    raise ValueError(f"Unexpected model size: {name}")
                digest.update(chunk)
                stream.write(chunk)
                if size - reported >= 32 * 1024**2:
                    print(f"  {name}: {size / record['size']:.0%}", flush=True)
                    reported = size
            stream.flush()
            os.fsync(stream.fileno())
        if size != record["size"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Model verification failed: {name}")
        temporary.replace(target)
        print(f"Downloaded and verified: {name}", flush=True)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Fetch all pinned model/encoder weights")
    parser.add_argument("--prompt-free", action="store_true", help="Fetch the four official prompt-free weights")
    parser.add_argument("--models", nargs="+", help="Select model basenames")
    parser.add_argument("--list", action="store_true", help="List the pinned weights; no download")
    parser.add_argument("--output", type=Path, default=root)
    args = parser.parse_args(argv)
    manifest = json.loads((root / "model_sources.json").read_text(encoding="utf-8"))
    by_name = {item["file"]: item for item in manifest["models"]}
    if args.list:
        for record in manifest["models"]:
            print(f"{record['file']} | {record['size']} bytes | SHA256 {record['sha256']}")
        return 0
    if sum(bool(value) for value in (args.all, args.models, args.prompt_free)) > 1:
        parser.error("Choose only one of --all, --models or --prompt-free")
    if args.prompt_free:
        selected = [name for name in by_name if name.endswith("-seg-pf.pt")]
    else:
        selected = list(by_name) if args.all else args.models or manifest["default_models"]
    if any(name not in by_name for name in selected):
        parser.error("Unknown model basename; see --list")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in selected:
        fetch(by_name[name], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
