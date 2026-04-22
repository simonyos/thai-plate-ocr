"""
Roboflow Universe dataset downloader.

Two public CC BY 4.0 datasets are used:

  DETECTOR  — dataset-format-conversion-iidaz / thailand-license-plate-recognition
             ~343 images, 1 class (plate), YOLO-formatted bounding boxes.

  RECOGNIZER — card-detector / thai-license-plate-character-detect
             ~2,500 images, 48 classes (Thai consonants + digits + province markers),
             YOLO-formatted bounding boxes per character.

Both are downloaded via the Roboflow Python SDK. The API key is read from
`ROBOFLOW_API_KEY` — never hard-coded.
"""

from __future__ import annotations

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
    workspace="dataset-format-conversion-iidaz",
    project="thailand-license-plate-recognition",
    version=1,
)

RECOGNIZER_REF = RoboflowDatasetRef(
    workspace="card-detector",
    project="thai-license-plate-character-detect",
    version=1,
)


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
