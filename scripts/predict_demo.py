import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from thai_plate_ocr.pipeline import PlatePipeline


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    det = root / "runs/detect/artifacts/detector/train/weights/best.pt"
    rec = root / "runs/detect/artifacts/recognizer/train/weights/best.pt"
    if not det.is_file():
        det = root / "artifacts/detector/train/weights/best.pt"
    if not rec.is_file():
        rec = root / "artifacts/recognizer/train/weights/best.pt"

    pipe = PlatePipeline(det, rec)

    candidates = list((root / "data/detector").rglob("*.jpg"))
    if not candidates:
        sys.exit(f"No candidate images under {root/'data/detector'}")
    random.seed(0)
    img_path = random.choice(candidates)
    img = Image.open(img_path).convert("RGB")
    preds = pipe.predict(img)

    print(f"image: {img_path}")
    for p in preds:
        print(f"  bbox={p.plate_bbox_xyxy} conf={p.plate_confidence:.3f} text={p.text!r}")

    draw = ImageDraw.Draw(img)
    for p in preds:
        draw.rectangle(p.plate_bbox_xyxy, outline="red", width=3)
        y = max(0, p.plate_bbox_xyxy[1] - 18)
        draw.text((p.plate_bbox_xyxy[0], y), p.text, fill="red")

    out = root / "reports/figures/sample_prediction.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
