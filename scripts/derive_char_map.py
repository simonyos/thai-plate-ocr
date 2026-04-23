"""Run the recognizer on a chart of known Thai consonants/digits and emit a
mapping from A## class code → Thai glyph.

The chart image contains 44 Thai consonants in canonical order
(ก ข ฃ ค ฅ ฆ ง จ ฉ | ช ซ ฌ ญ ฎ ฏ ฐ ฑ ฒ | ณ ด ต ถ ท ธ น บ ป |
 ผ ฝ พ ฟ ภ ม ย ร ล | ว ศ ษ ส ห ฬ อ ฮ).

We run the YOLO recognizer on it, cluster detections by y-centre into rows
(same algorithm the pipeline uses for two-line plates), sort each row
left-to-right, and pair each detection with the expected glyph.

Usage:
    python scripts/derive_char_map.py \\
        --weights runs/detect/artifacts/recognizer/train/weights/best.pt \\
        --chart scripts/assets/thai_consonants_chart.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from thai_plate_ocr.pipeline import CharBox, order_characters

EXPECTED_ROWS: list[list[str]] = [
    ["ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ"],
    ["ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", "ฑ", "ฒ"],
    ["ณ", "ด", "ต", "ถ", "ท", "ธ", "น", "บ", "ป"],
    ["ผ", "ฝ", "พ", "ฟ", "ภ", "ม", "ย", "ร", "ล"],
    ["ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"],
]


def _recognize(weights: Path, chart: Path, conf: float, imgsz: int) -> list[CharBox]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.predict(source=str(chart), conf=conf, imgsz=imgsz, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    out: list[CharBox] = []
    for cid, cv, box in zip(cls_ids, confs, xyxy, strict=False):
        out.append(
            CharBox(
                cls=str(result.names[int(cid)]),
                conf=float(cv),
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out-json", type=Path, default=Path("src/thai_plate_ocr/char_map.json"))
    ap.add_argument("--annotated-out", type=Path, default=Path("reports/figures/char_map_debug.png"))
    args = ap.parse_args()

    chars = _recognize(args.weights, args.chart, args.conf, args.imgsz)
    print(f"raw detections: {len(chars)}")

    rows = order_characters(chars, line_gap_frac=0.4)
    for i, row in enumerate(rows):
        print(f"row {i} ({len(row)} dets): {[c.cls for c in row]}")

    # Pair detections with expected glyphs.
    if len(rows) != len(EXPECTED_ROWS):
        print(f"WARNING: detected {len(rows)} rows, expected {len(EXPECTED_ROWS)}")

    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for det_row, exp_row in zip(rows, EXPECTED_ROWS, strict=False):
        if len(det_row) != len(exp_row):
            print(f"WARNING: row len mismatch: det={len(det_row)}, expected={len(exp_row)}")
        for det, glyph in zip(det_row, exp_row, strict=False):
            if det.cls in mapping and mapping[det.cls] != glyph:
                print(f"CONFLICT: {det.cls} was {mapping[det.cls]!r}, now {glyph!r} (conf {det.conf:.2f})")
            mapping[det.cls] = glyph
        # note any extras
        if len(det_row) > len(exp_row):
            unmapped.extend(d.cls for d in det_row[len(exp_row):])

    print("\n=== MAPPING ===")
    for k in sorted(mapping):
        print(f"  {k} → {mapping[k]}")
    print(f"\ntotal mapped consonants: {len(mapping)}")
    if unmapped:
        print(f"extra detections (unmapped): {unmapped}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    print(f"\nWrote {args.out_json}")

    # Draw debug image.
    img = Image.open(args.chart).convert("RGB")
    draw = ImageDraw.Draw(img)
    for c in chars:
        draw.rectangle([c.x1, c.y1, c.x2, c.y2], outline="red", width=2)
        draw.text((c.x1, max(0, c.y1 - 14)), f"{c.cls}:{mapping.get(c.cls, '?')}", fill="red")
    args.annotated_out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.annotated_out)
    print(f"Wrote {args.annotated_out}")


if __name__ == "__main__":
    main()
