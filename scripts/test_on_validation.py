"""Run the end-to-end pipeline on the detector validation split.

Emits:
  reports/figures/test_gallery.png            grid of N annotated images
  reports/test_predictions.md                 one row per image with pred string
  reports/figures/test_<idx>.png              individual annotated crops
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from thai_plate_ocr.pipeline import PlatePipeline

N_IMAGES = 12
SEED = 7


def _find_weights(root: Path, stage: str) -> Path:
    candidates = [
        root / f"runs/detect/artifacts/{stage}/train/weights/best.pt",
        root / f"artifacts/{stage}/train/weights/best.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No {stage} weights under {root}")


def _find_val_images(root: Path) -> list[Path]:
    val_dir = root / "data/detector/valid/images"
    if not val_dir.is_dir():
        val_dir = root / "data/detector/val/images"
    if not val_dir.is_dir():
        val_dir = root / "data/detector/test/images"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"No val/valid/test images dir under {root/'data/detector'}")
    return sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))


def _annotate(img: Image.Image, preds: list) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for p in preds:
        draw.rectangle(p.plate_bbox_xyxy, outline="red", width=4)
        y = max(0, p.plate_bbox_xyxy[1] - 24)
        label = f"{p.text}  ({p.plate_confidence:.2f})"
        draw.text((p.plate_bbox_xyxy[0], y), label, fill="red", font=font)
    return out


def _make_gallery(images: list[Image.Image], out: Path, cols: int = 3) -> None:
    if not images:
        return
    thumb_w, thumb_h = 480, 320
    resized = []
    for im in images:
        cp = im.copy()
        cp.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "black")
        canvas.paste(cp, ((thumb_w - cp.width) // 2, (thumb_h - cp.height) // 2))
        resized.append(canvas)
    rows = math.ceil(len(resized) / cols)
    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "black")
    for i, im in enumerate(resized):
        r, c = divmod(i, cols)
        grid.paste(im, (c * thumb_w, r * thumb_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    det = _find_weights(root, "detector")
    rec = _find_weights(root, "recognizer")
    pipe = PlatePipeline(det, rec)

    candidates = _find_val_images(root)
    random.seed(SEED)
    chosen = random.sample(candidates, min(N_IMAGES, len(candidates)))

    figs_dir = root / "reports/figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    gallery = []
    for i, p in enumerate(chosen):
        img = Image.open(p).convert("RGB")
        preds = pipe.predict(img)
        annotated = _annotate(img, preds)
        annotated.save(figs_dir / f"test_{i:02d}.png")
        gallery.append(annotated)
        if preds:
            row = {
                "idx": i,
                "file": p.name,
                "n_plates": len(preds),
                "best_conf": max(q.plate_confidence for q in preds),
                "texts": [q.text for q in preds],
            }
        else:
            row = {"idx": i, "file": p.name, "n_plates": 0, "best_conf": 0.0, "texts": []}
        print(f"[{i:02d}] {p.name}: {row['n_plates']} plate(s), conf={row['best_conf']:.3f}, text={row['texts']}")
        rows.append(row)

    _make_gallery(gallery, figs_dir / "test_gallery.png", cols=3)

    md = ["# Test predictions — detector validation split", ""]
    md.append("Each row is one held-out image. `texts` is the predicted plate string(s)")
    md.append("(characters are emitted as Roboflow class codes A01..A54 — see report for mapping).")
    md.append("")
    md.append("| idx | file | plates | conf | texts |")
    md.append("|---:|---|---:|---:|---|")
    for r in rows:
        texts_cell = " / ".join(r["texts"]) if r["texts"] else "_(none)_"
        md.append(f"| {r['idx']} | `{r['file']}` | {r['n_plates']} | {r['best_conf']:.3f} | {texts_cell} |")

    out_md = root / "reports/test_predictions.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    (root / "reports/test_predictions.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_md}")
    print(f"Wrote {figs_dir/'test_gallery.png'}")


if __name__ == "__main__":
    main()
