"""
FastAPI inference endpoint. POST an image to /predict; get back a list of detected
plates, each with bounding box, recognised text lines, and per-character boxes.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from thai_plate_ocr.config import Settings
from thai_plate_ocr.pipeline import PlatePipeline


class CharacterOut(BaseModel):
    cls: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float]


class PlateOut(BaseModel):
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    text: str
    text_lines: list[str]
    characters: list[CharacterOut]


class PredictionOut(BaseModel):
    plates: list[PlateOut]


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def _pipeline() -> PlatePipeline:
    s = _settings()
    det = s.detector_runs_dir / "train" / "weights" / "best.pt"
    rec = s.recognizer_runs_dir / "train" / "weights" / "best.pt"
    if not det.exists() or not rec.exists():
        raise RuntimeError(
            "Trained weights not found. Run `plate train-detector` and "
            "`plate train-recognizer` first."
        )
    return PlatePipeline(det, rec)


app = FastAPI(title="Thai License Plate OCR API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
async def predict(image: UploadFile = File(...)) -> PredictionOut:  # noqa: B008
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")
    data = await image.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    results = _pipeline().predict(pil)
    plates: list[PlateOut] = []
    for r in results:
        plates.append(
            PlateOut(
                bbox_xyxy=r.plate_bbox_xyxy,
                confidence=r.plate_confidence,
                text=r.text,
                text_lines=r.text_lines,
                characters=[
                    CharacterOut(
                        cls=c.cls,
                        conf=c.conf,
                        bbox_xyxy=(c.x1, c.y1, c.x2, c.y2),
                    )
                    for c in r.characters
                ],
            )
        )
    return PredictionOut(plates=plates)


__all__: list[Any] = ["app"]
