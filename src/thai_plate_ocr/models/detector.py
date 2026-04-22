"""
Stage-1 model: a YOLOv8 plate detector fine-tuned on a Thai plate dataset.

The Ultralytics YOLO wrapper handles training, checkpointing, and eval by itself;
this module is a thin adapter that reads from our `Settings` and logs the training
configuration to MLflow.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow

from thai_plate_ocr.config import Settings


@dataclass
class DetectorConfig:
    epochs: int = 40
    imgsz: int = 640
    batch: int = 16
    patience: int = 10


@dataclass
class DetectorResult:
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    best_weights: Path
    train_seconds: float


def run(settings: Settings, cfg: DetectorConfig | None = None) -> DetectorResult:
    from ultralytics import YOLO

    cfg = cfg or DetectorConfig()
    data_yaml = settings.detector_dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run `plate download` first."
        )

    settings.artifacts_root.mkdir(parents=True, exist_ok=True)
    project_dir = settings.detector_runs_dir
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(settings.detector_weights)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("thai-plate-ocr/detector")

    with mlflow.start_run(run_name=settings.detector_weights):
        mlflow.log_params({
            "base_weights": settings.detector_weights,
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "patience": cfg.patience,
            "data": str(data_yaml),
        })

        t0 = time.perf_counter()
        results = model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            patience=cfg.patience,
            project=str(project_dir),
            name="train",
            exist_ok=True,
            device=None if settings.device == "auto" else settings.device,
            seed=settings.seed,
        )
        train_seconds = time.perf_counter() - t0

        metrics = model.val(data=str(data_yaml))
        payload = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(payload)

        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        mlflow.log_artifact(str(best_weights))

    (settings.artifacts_root / "detector_summary.json").write_text(json.dumps(payload, indent=2))

    return DetectorResult(
        mAP50=payload["mAP50"],
        mAP50_95=payload["mAP50_95"],
        precision=payload["precision"],
        recall=payload["recall"],
        best_weights=best_weights,
        train_seconds=train_seconds,
    )
