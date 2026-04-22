"""
End-to-end inference pipeline: image → plate crops → ordered character strings.

Two-line Thai plate handling
----------------------------
Thai car plates often render the province name on a second line below the
registration. Rather than train a separate two-line classifier, we cluster
character detections by their y-centre using a simple 1-D density split:

  1. Collect all character centres (cx, cy).
  2. Compute vertical gaps; if any gap exceeds `line_gap_frac` of the plate
     height, interpret it as a line break.
  3. Within each line, sort detections left-to-right by x-centre.

This is robust to minor skew and avoids reliance on any particular plate
layout spec.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CharBox:
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


@dataclass
class PlatePrediction:
    plate_bbox_xyxy: tuple[float, float, float, float]
    plate_confidence: float
    text_lines: list[str]
    characters: list[CharBox]

    @property
    def text(self) -> str:
        return " ".join(self.text_lines)


def order_characters(
    chars: Sequence[CharBox], line_gap_frac: float = 0.6
) -> list[list[CharBox]]:
    """Cluster character detections into lines, then sort each line left-to-right.

    A new line is started whenever the vertical gap between successive y-centres
    (sorted ascending) exceeds `line_gap_frac` times the median character height.
    """
    if not chars:
        return []

    ordered = sorted(chars, key=lambda c: c.cy)
    median_h = float(np.median([c.height for c in ordered]))
    threshold = line_gap_frac * median_h

    lines: list[list[CharBox]] = [[ordered[0]]]
    for prev, curr in pairwise(ordered):
        gap = curr.cy - prev.cy
        if gap > threshold:
            lines.append([curr])
        else:
            lines[-1].append(curr)

    return [sorted(line, key=lambda c: c.cx) for line in lines]


class PlatePipeline:
    def __init__(
        self,
        detector_weights: Path | str,
        recognizer_weights: Path | str,
        detector_conf: float = 0.25,
        recognizer_conf: float = 0.25,
        detector_imgsz: int = 640,
        recognizer_imgsz: int = 480,
    ) -> None:
        from ultralytics import YOLO

        self.detector = YOLO(str(detector_weights))
        self.recognizer = YOLO(str(recognizer_weights))
        self.detector_conf = detector_conf
        self.recognizer_conf = recognizer_conf
        self.detector_imgsz = detector_imgsz
        self.recognizer_imgsz = recognizer_imgsz

    def _recognize(self, crop: Image.Image) -> list[CharBox]:
        result = self.recognizer.predict(
            source=crop,
            conf=self.recognizer_conf,
            imgsz=self.recognizer_imgsz,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        conf_vals = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        chars: list[CharBox] = []
        for cid, cv, box in zip(cls_ids, conf_vals, xyxy, strict=False):
            chars.append(
                CharBox(
                    cls=str(result.names[int(cid)]),
                    conf=float(cv),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
            )
        return chars

    def predict(self, image: Image.Image | str | Path) -> list[PlatePrediction]:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        detections = self.detector.predict(
            source=image,
            conf=self.detector_conf,
            imgsz=self.detector_imgsz,
            verbose=False,
        )[0]

        results: list[PlatePrediction] = []
        if detections.boxes is None or len(detections.boxes) == 0:
            return results

        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs, strict=False):
            x1, y1, x2, y2 = (int(max(0, v)) for v in box)
            crop = image.crop((x1, y1, x2, y2))
            chars = self._recognize(crop)
            lines = order_characters(chars)
            text_lines = ["".join(c.cls for c in line) for line in lines]
            results.append(
                PlatePrediction(
                    plate_bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    plate_confidence=float(conf),
                    text_lines=text_lines,
                    characters=chars,
                )
            )
        return results
