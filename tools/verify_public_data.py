from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "DATA_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
            continue
        frame = pd.read_csv(path, low_memory=False)
        checks = {
            "rows": len(frame) == int(item["rows"]),
            "columns": len(frame.columns) == int(item["columns"]),
            "sha256": sha256(path) == item["sha256"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append(f"{item['path']}: {', '.join(failed)}")
        else:
            print(
                f"OK {item['path']}: {len(frame)} rows, "
                f"{len(frame.columns)} columns"
            )
    if failures:
        raise SystemExit("Data-manifest verification failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
