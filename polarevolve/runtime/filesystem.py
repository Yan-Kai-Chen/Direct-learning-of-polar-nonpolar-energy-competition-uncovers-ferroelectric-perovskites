"""UTF-8 input and atomic runtime file transactions."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield physical line numbers and decoded values; consumers own their schemas."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = asdict(payload) if is_dataclass(payload) else payload
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


__all__ = ["atomic_json", "load_json", "iter_jsonl"]
