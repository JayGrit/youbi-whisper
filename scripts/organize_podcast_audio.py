#!/usr/bin/env python3
"""Collect existing podcast NotebookLM audio into one Desktop audio directory."""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

DESKTOP = Path("/Users/hoshuuch/Desktop")
DOWNLOADS = Path("/Users/hoshuuch/Downloads")
OUTPUT = DESKTOP / "podcast_all_notebooklm"
DESTINATION = OUTPUT / "audio"
NINE = DESKTOP / "podcast_relabel_9" / "audio"
PREVIOUS = DESKTOP / "podcast_remaining_notebooklm" / "audio"
MISSING_BEFORE = (58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
                  75, 76, 78, 79, 81, 83, 84, 94, 98, 99, 100, 101)


def link_existing(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def collect_from_zip(task_id: str, destination: Path) -> dict | None:
    directories = [directory for directory in DOWNLOADS.iterdir()
                   if directory.is_dir() and re.fullmatch(re.escape(task_id) + r"(?:\(\d+\))?", directory.name)]
    archives = [archive for directory in directories for archive in sorted(directory.glob("*.zip"))]
    archives.extend(sorted(DOWNLOADS.glob(f"{task_id}*.zip")))
    # podcast-81 has two ZIPs; the (1) file matches the recorded AliDrive archive.
    if task_id == "podcast-81":
        archives.sort(key=lambda item: item.name != "podcast-81_process_assets(1).zip")
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            members = [item for item in bundle.infolist()
                       if not item.is_dir() and item.filename.startswith(f"{task_id}/notebooklm/")
                       and item.filename.lower().endswith(".m4a")]
            if len(members) != 1:
                continue
            if bundle.testzip() is not None:
                raise RuntimeError(f"{archive}: ZIP integrity check failed")
            temporary = destination.with_name(destination.name + ".part")
            try:
                with bundle.open(members[0]) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if temporary.stat().st_size != members[0].file_size:
                    raise RuntimeError(f"{archive}: extracted size mismatch")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            return {"source": str(archive), "member": members[0].filename}
    # Some ZIPs have already been unpacked manually.
    extracted = [audio for directory in directories for audio in directory.rglob("*.m4a")]
    extracted.extend(DOWNLOADS.glob(f"{task_id}*.m4a"))
    if len(extracted) == 1:
        link_existing(extracted[0], destination)
        return {"source": str(extracted[0])}
    return None


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for item in PREVIOUS.glob("podcast-*/*.m4a"):
        sources[item.parent.name] = item
    for item in NINE.glob("podcast-*.m4a"):
        sources[item.stem] = item
    task_ids = set(sources) | {f"podcast-{number}" for number in MISSING_BEFORE}
    report: dict[str, dict] = {}
    for task_id in sorted(task_ids, key=lambda value: int(value.split("-", 1)[1])):
        destination = DESTINATION / f"{task_id}.m4a"
        if destination.is_file() and destination.stat().st_size > 0:
            report[task_id] = {"status": "present", "path": str(destination), "bytes": destination.stat().st_size}
            continue
        source = sources.get(task_id)
        if source:
            link_existing(source, destination)
            origin = {"source": str(source)}
        else:
            origin = collect_from_zip(task_id, destination)
        if origin is None:
            report[task_id] = {"status": "missing"}
        else:
            report[task_id] = {"status": "ready", "path": str(destination),
                               "bytes": destination.stat().st_size, **origin}
        print(f"{task_id}: {report[task_id]['status']}", flush=True)
    (OUTPUT / "audio_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ready = sum(row["status"] in {"ready", "present"} for row in report.values())
    print(f"Ready {ready}/{len(report)} audio files in {DESTINATION}")


if __name__ == "__main__":
    main()
