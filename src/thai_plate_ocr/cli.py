"""Typer CLI entry points."""

from __future__ import annotations

import typer
from rich import print as rprint

from thai_plate_ocr.config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Thai plate OCR CLI.")


@app.command("download")
def download() -> None:
    """Download both Roboflow datasets (plate detector + character recognizer)."""
    from thai_plate_ocr.data.download import download_all

    paths = download_all(Settings())
    rprint({k: str(v) for k, v in paths.items()})


@app.command("train-detector")
def train_detector() -> None:
    """Fine-tune YOLOv8 on the Thai plate-detection dataset (stage 1)."""
    from thai_plate_ocr.models.detector import run

    r = run(Settings())
    rprint({
        "mAP50": r.mAP50,
        "mAP50_95": r.mAP50_95,
        "precision": r.precision,
        "recall": r.recall,
        "best_weights": str(r.best_weights),
    })


@app.command("train-recognizer")
def train_recognizer() -> None:
    """Fine-tune YOLOv8 on the Thai plate character dataset (stage 2)."""
    from thai_plate_ocr.models.recognizer import run

    r = run(Settings())
    rprint({
        "mAP50": r.mAP50,
        "mAP50_95": r.mAP50_95,
        "precision": r.precision,
        "recall": r.recall,
        "best_weights": str(r.best_weights),
    })


@app.command("evaluate")
def evaluate() -> None:
    """Aggregate per-stage metrics into reports/summary.md + a bar chart."""
    from thai_plate_ocr.evaluate import run

    out = run(Settings())
    rprint(f"Wrote [bold]{out}[/bold]")


@app.command("predict")
def predict(image_path: str) -> None:
    """Run the end-to-end pipeline on one image path."""
    from thai_plate_ocr.pipeline import PlatePipeline

    s = Settings()
    det = s.detector_runs_dir / "train" / "weights" / "best.pt"
    rec = s.recognizer_runs_dir / "train" / "weights" / "best.pt"
    pipe = PlatePipeline(det, rec)
    for i, plate in enumerate(pipe.predict(image_path)):
        rprint({
            "plate_index": i,
            "bbox_xyxy": plate.plate_bbox_xyxy,
            "confidence": plate.plate_confidence,
            "text": plate.text,
            "text_lines": plate.text_lines,
        })


if __name__ == "__main__":
    app()
