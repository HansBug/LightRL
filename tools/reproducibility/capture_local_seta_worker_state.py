#!/usr/bin/env python3
"""Capture a credential-free source snapshot for an in-RJob SETA worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOTS = ("agentic_rl", "deploy/workers")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".log", ".pid"}
SENSITIVE_NAMES = {".env", "id_rsa", "id_ed25519"}
SENSITIVE_CONTENT = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\b(?:sk|ghp|glpat)[-_][A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"WANDB_API_KEY\s*=\s*[\"']?[A-Za-z0-9]{20,}"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in SOURCE_ROOTS:
        for path in (repo / root_name).rglob("*"):
            relative = path.relative_to(repo)
            if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES or path.name in SENSITIVE_NAMES:
                continue
            lowered = path.name.lower()
            if lowered.endswith((".pem", ".key")) or "secret" in lowered:
                continue
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(repo).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-url", required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = source_files(repo)
    if not files:
        raise SystemExit("no worker source files selected")
    sensitive_hits = [
        path.relative_to(repo).as_posix()
        for path in files
        if any(pattern.search(path.read_bytes()) for pattern in SENSITIVE_CONTENT)
    ]
    if sensitive_hits:
        raise SystemExit(f"refusing to archive files with credential-like content: {sensitive_hits}")

    sha_manifest = output / "source.sha256"
    with sha_manifest.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{sha256_file(path)}  {path.relative_to(repo).as_posix()}\n")

    archive = output / "worker-source.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(repo), recursive=False)

    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "worker_url": args.worker_url,
        "source_roots": list(SOURCE_ROOTS),
        "source_file_count": len(files),
        "source_hash_errors": 0,
        "source_archive": archive.name,
        "source_archive_sha256": sha256_file(archive),
        "source_sha256_manifest": sha_manifest.name,
        "source_sha256_manifest_sha256": sha256_file(sha_manifest),
        "secrets_included": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
