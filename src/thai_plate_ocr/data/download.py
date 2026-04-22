"""
Roboflow Universe dataset downloader.

Two public CC BY 4.0 datasets are used:

  DETECTOR   — nextra / thai-licence-plate-detect-b93xq
              ~294 images, 1 class (`th-plate`), YOLO-formatted plate bounding boxes.

  RECOGNIZER — card-detector / thai-license-plate-character-detect
              ~2,521 images, 46 classes (A01..A54, Thai consonants + digits + province
              markers), YOLO-formatted bounding boxes per character.

Both are downloaded via the Roboflow Python SDK. The API key is read from
`ROBOFLOW_API_KEY` — never hard-coded.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from thai_plate_ocr.config import Settings


@dataclass(frozen=True)
class RoboflowDatasetRef:
    workspace: str
    project: str
    version: int
    export_format: str = "yolov8"


DETECTOR_REF = RoboflowDatasetRef(
    workspace="nextra",
    project="thai-licence-plate-detect-b93xq",
    version=1,
)

RECOGNIZER_REF = RoboflowDatasetRef(
    workspace="card-detector",
    project="thai-license-plate-character-detect",
    version=1,
)


def _flatten_if_nested(dest: Path) -> None:
    """Roboflow sometimes extracts into `dest/<project-name>/` instead of `dest/`.

    If we see exactly one subdirectory with a data.yaml in it and no data.yaml at the
    top level, move the contents up one level.
    """
    if (dest / "data.yaml").is_file():
        return
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    nested = next((p for p in subdirs if (p / "data.yaml").is_file()), None)
    if nested is None:
        return
    for child in nested.iterdir():
        shutil.move(str(child), str(dest / child.name))
    nested.rmdir()


def _fetch(ref: RoboflowDatasetRef, dest: Path, api_key: str) -> Path:
    from roboflow import Roboflow

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Export it in your shell or put it in .env."
        )

    dest.mkdir(parents=True, exist_ok=True)
    data_yaml = dest / "data.yaml"
    if data_yaml.is_file():
        return dest

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ref.workspace).project(ref.project)
    project.version(ref.version).download(ref.export_format, location=str(dest))

    _flatten_if_nested(dest)

    if not data_yaml.is_file():
        contents = sorted(str(p.relative_to(dest)) for p in dest.rglob("*"))[:40]
        raise RuntimeError(
            f"Roboflow download for {ref.workspace}/{ref.project} did not produce "
            f"{data_yaml}. Contents of {dest} after download: {contents}"
        )
    return dest


def download_detector(settings: Settings) -> Path:
    settings.ensure_dirs()
    return _fetch(DETECTOR_REF, settings.detector_dataset_dir, settings.roboflow_api_key)


def download_recognizer(settings: Settings) -> Path:
    settings.ensure_dirs()
    return _fetch(RECOGNIZER_REF, settings.recognizer_dataset_dir, settings.roboflow_api_key)


def download_all(settings: Settings) -> dict[str, Path]:
    return {
        "detector": download_detector(settings),
        "recognizer": download_recognizer(settings),
    }
