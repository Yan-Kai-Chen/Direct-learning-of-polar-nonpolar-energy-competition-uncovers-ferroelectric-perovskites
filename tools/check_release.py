"""Check only this Git checkout for unintended public-release contents."""

from __future__ import annotations

import re
import subprocess
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pyc", ".pt", ".pth", ".ckpt", ".safetensors", ".joblib", ".pkl", ".npy", ".npz", ".log"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".cff", ".txt"}
PATTERNS = {
    "GitHub credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "API credential": re.compile(r"\bsk-[A-Za-z0-9_-]{30,}\b"),
    "user profile path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/(?:Users|home)/[A-Za-z0-9_.-]+/|/root[/]"),
}


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True,
    )
    paths = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    failures = []
    checked = 0
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue
        checked += 1
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"unintended artifact: {relative}")
        if path.suffix.lower() == ".cif" and not relative.startswith("examples/"):
            failures.append(f"unexpected structure: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
            continue
        content = path.read_text(encoding="utf-8")
        for name, pattern in PATTERNS.items():
            match = pattern.search(content)
            if match:
                line = content.count("\n", 0, match.start()) + 1
                failures.append(f"{name}: {relative}:{line}")
    if failures:
        raise SystemExit("Release check failed:\n" + "\n".join(failures))
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["diffusion_files"]:
        relative = Path(record["destination"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("Unsafe source-manifest path")
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if digest != record.get("published_sha256"):
            raise SystemExit(f"Published source hash mismatch: {relative}")
    print(f"Public-release content check OK: {checked} files")


if __name__ == "__main__":
    main()
