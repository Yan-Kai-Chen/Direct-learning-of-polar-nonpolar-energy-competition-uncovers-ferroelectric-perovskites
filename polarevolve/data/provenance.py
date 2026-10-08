"""Deterministic source identity for training and sampling evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_FINGERPRINT_CONTRACT = "gt_sge_source_fingerprint_v1"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_fingerprint(root: Path | None = None) -> dict[str, object]:
    resolved = (root or repository_root()).expanduser().resolve()
    files = sorted((resolved / "src" / "polarevolve").rglob("*.py"))
    files.extend(sorted((resolved / "bash" / "version9").glob("*.sh")))
    files.extend(
        path
        for path in (
            resolved / "pyproject.toml",
            resolved / "environment.cluster.yml",
        )
        if path.is_file()
    )
    unique_files = sorted(set(files))
    digest = hashlib.sha256()
    for path in unique_files:
        digest.update(path.relative_to(resolved).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "contract": SOURCE_FINGERPRINT_CONTRACT,
        "method": "sha256_path_and_content_v1",
        "sha256": digest.hexdigest(),
        "files": len(unique_files),
    }


__all__ = ["SOURCE_FINGERPRINT_CONTRACT", "repository_root", "source_fingerprint"]
