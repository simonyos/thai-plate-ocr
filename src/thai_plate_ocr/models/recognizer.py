"""
Stage-2 model: a YOLOv8 character-level detector fine-tuned on the Thai plate
character dataset (~48 classes = Thai consonants + 0-9 digits + province markers).

At inference we run this on cropped plates produced by the Stage-1 detector.
Character bounding boxes are then spatially ordered (top-bottom lines, left-right
within a line) by `thai_plate_ocr.pipeline.order_characters`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow

from thai_plate_ocr.config import Settings


@dataclass
class RecognizerConfig:
    epochs: int = 60
    imgsz: int = 480
    batch: int = 32
    patience: int = 15


@dataclass
class RecognizerResult:
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    best_weights: Path
    train_seconds: float
    classes: list[str]


def run(settings: Settings, cfg: RecognizerConfig | None = None) -> RecognizerResult:
    import yaml
    from ultralytics import YOLO

    cfg = cfg or RecognizerConfig()
    data_yaml = settings.recognizer_dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run `plate download` first."
        )

    settings.artifacts_root.mkdir(parents=True, exist_ok=True)
    project_dir = settings.recognizer_runs_dir
    project_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml, encoding="utf-8") as f:
        class_names = yaml.safe_load(f).get("names", [])

    model = YOLO(settings.recognizer_weights)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("thai-plate-ocr/recognizer")

    with mlflow.start_run(run_name=settings.recognizer_weights):
        mlflow.log_params({
            "base_weights": settings.recognizer_weights,
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "patience": cfg.patience,
            "data": str(data_yaml),
            "num_classes": len(class_names),
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
            "classes": class_names,
        }
        mlflow.log_metrics({k: v for k, v in payload.items() if isinstance(v, int | float)})

        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        mlflow.log_artifact(str(best_weights))

    (settings.artifacts_root / "recognizer_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )

    return RecognizerResult(
        mAP50=payload["mAP50"],
        mAP50_95=payload["mAP50_95"],
        precision=payload["precision"],
        recall=payload["recall"],
        best_weights=best_weights,
        train_seconds=train_seconds,
        classes=class_names,
    )
