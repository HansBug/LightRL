#!/usr/bin/env python3
"""Build a credential-free Ubuntu Jammy Docker offline APT repository.

The login environment has working curl egress while isolated RJob Pods do not.
This tool downloads signed package-index payloads, resolves the dependency
closure, verifies every package SHA256, and emits a local ``Packages`` index.
The RJob installs only the requested roots and actually missing dependencies.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import lzma
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import apt_pkg


DEFAULT_MIRROR = "http://mirrors.aliyun.com/ubuntu"
SUITES = ("jammy", "jammy-updates", "jammy-security")
COMPONENTS = ("main", "universe")
ROOT_CANDIDATES = (
    "docker.io",
    "fuse-overlayfs",
    "iproute2",
    "iptables",
    "slirp4netns",
    "uidmap",
    "ca-certificates",
)


def curl(url: str, output: Path) -> None:
    subprocess.run(
        ["curl", "-fsSL", "--retry", "4", "--retry-delay", "2", "-o", str(output), url],
        check=True,
    )


def parse_stanzas(text: str):
    stanza: dict[str, str] = {}
    current = ""
    for line in text.splitlines() + [""]:
        if not line:
            if stanza:
                yield stanza
            stanza = {}
            current = ""
        elif line[0].isspace() and current:
            stanza[current] += " " + line.strip()
        else:
            current, value = line.split(":", 1)
            stanza[current] = value.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_newer(catalog: dict[str, dict[str, str]], stanza: dict[str, str]) -> None:
    package = stanza.get("Package", "")
    version = stanza.get("Version", "")
    if not package or not version or not stanza.get("Filename") or not stanza.get("SHA256"):
        return
    if stanza.get("Architecture") not in {"amd64", "all"}:
        return
    current = catalog.get(package)
    if current is None or apt_pkg.version_compare(version, current["Version"]) > 0:
        catalog[package] = stanza


def dependencies(stanza: dict[str, str]) -> list[list[str]]:
    groups: list[list[str]] = []
    for field in ("Pre-Depends", "Depends"):
        value = stanza.get(field, "")
        if not value:
            continue
        for alternatives in apt_pkg.parse_depends(value, strip_multi_arch=True):
            groups.append([item[0] for item in alternatives])
    return groups


def dependency_closure(catalog: dict[str, dict[str, str]], roots: list[str]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    pending = list(roots)
    while pending:
        package = pending.pop()
        if package in selected:
            continue
        stanza = catalog.get(package)
        if stanza is None:
            raise RuntimeError(f"package missing from Jammy indexes: {package}")
        selected[package] = stanza
        for alternatives in dependencies(stanza):
            candidate = next((name for name in alternatives if name in catalog), None)
            # Virtual/essential dependencies absent from the downloaded main +
            # universe catalog are satisfied by the RFT base image.
            if candidate and candidate not in selected:
                pending.append(candidate)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)
    index_dir = output / ".indexes"
    index_dir.mkdir()

    apt_pkg.init_system()
    catalog: dict[str, dict[str, str]] = {}
    for suite in SUITES:
        for component in COMPONENTS:
            name = f"{suite}-{component}-Packages.xz"
            path = index_dir / name
            url = f"{args.mirror}/dists/{suite}/{component}/binary-amd64/Packages.xz"
            print(f"index {suite}/{component}", flush=True)
            curl(url, path)
            with lzma.open(path, "rt", encoding="utf-8", errors="replace") as stream:
                for stanza in parse_stanzas(stream.read()):
                    choose_newer(catalog, stanza)

    compose = next((name for name in ("docker-compose-v2", "docker-compose-plugin") if name in catalog), None)
    if compose is None:
        raise SystemExit("Jammy indexes provide neither docker-compose-v2 nor docker-compose-plugin")
    roots = [*ROOT_CANDIDATES, compose]
    selected = dependency_closure(catalog, roots)
    print(f"resolved {len(selected)} packages; compose={compose}", flush=True)

    def download(item: tuple[str, dict[str, str]]) -> tuple[str, str]:
        package, stanza = item
        filename = Path(stanza["Filename"]).name
        destination = output / filename
        curl(f"{args.mirror}/{stanza['Filename']}", destination)
        actual = sha256_file(destination)
        if actual != stanza["SHA256"]:
            raise RuntimeError(f"SHA256 mismatch for {package}: {actual} != {stanza['SHA256']}")
        return package, filename

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        downloaded = dict(executor.map(download, sorted(selected.items())))

    shutil.rmtree(index_dir)
    (output / "ROOT_PACKAGES").write_text("\n".join(roots) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in sorted(output.glob("*.deb")):
            stream.write(f"{sha256_file(path)}  {path.name}\n")
    packages = subprocess.run(
        ["dpkg-scanpackages", ".", "/dev/null"],
        cwd=output,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output / "Packages").write_text(packages, encoding="utf-8")
    with gzip.GzipFile(filename=str(output / "Packages.gz"), mode="wb", mtime=0) as stream:
        stream.write(packages.encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "distribution": "Ubuntu 22.04 Jammy amd64",
        "mirror": args.mirror,
        "suites": list(SUITES),
        "components": list(COMPONENTS),
        "root_packages": roots,
        "package_count": len(downloaded),
        "packages_sha256": sha256_file(output / "Packages"),
        "credential_free": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"bundle ready: {output} ({len(downloaded)} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
