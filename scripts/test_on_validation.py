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

from thai_plate_ocr.char_map import to_glyph, translate
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


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer a font that covers Thai glyphs.
    for path in (
        "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
        "/usr/share/fonts/truetype/tlwg/Garuda-Bold.ttf",
        "/usr/share/fonts/truetype/tlwg/Loma-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Ayuthaya.ttf",
        "/System/Library/Fonts/Thonburi.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate(img: Image.Image, preds: list) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _load_font(28)
    char_font = _load_font(20)
    for p in preds:
        draw.rectangle(p.plate_bbox_xyxy, outline="red", width=6)
        text_thai = translate([c.cls for c in p.characters])
        label = f"{text_thai}  ({p.plate_confidence:.2f})"
        tx = p.plate_bbox_xyxy[0]
        ty = max(0, p.plate_bbox_xyxy[1] - 38)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((tx, ty), label, fill="white", font=font)
        # per-character boxes (translated to original-image coords)
        px, py = p.plate_bbox_xyxy[0], p.plate_bbox_xyxy[1]
        for c in p.characters:
            cx1, cy1 = c.x1 + px, c.y1 + py
            cx2, cy2 = c.x2 + px, c.y2 + py
            draw.rectangle([cx1, cy1, cx2, cy2], outline="lime", width=2)
            glyph = to_glyph(c.cls)
            # label above the char box
            lb = draw.textbbox((cx1, max(0, cy1 - 22)), glyph, font=char_font)
            draw.rectangle(lb, fill="lime")
            draw.text((cx1, max(0, cy1 - 22)), glyph, fill="black", font=char_font)
    return out


def _zoomed_crop(img: Image.Image, preds: list, pad: float = 0.6, min_w: int = 640) -> Image.Image | None:
    """Crop around the highest-confidence plate with padding, scale to min_w wide."""
    if not preds:
        return None
    p = max(preds, key=lambda q: q.plate_confidence)
    x1, y1, x2, y2 = p.plate_bbox_xyxy
    w, h = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - pad * w))
    cy1 = max(0, int(y1 - pad * h))
    cx2 = min(img.width, int(x2 + pad * w))
    cy2 = min(img.height, int(y2 + pad * h))
    crop = img.crop((cx1, cy1, cx2, cy2))
    if crop.width < min_w:
        scale = min_w / crop.width
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)))
    # re-draw bbox in crop coords
    sx = crop.width / (cx2 - cx1)
    sy = crop.height / (cy2 - cy1)
    draw = ImageDraw.Draw(crop)
    font = _load_font(32)
    bx1 = (x1 - cx1) * sx
    by1 = (y1 - cy1) * sy
    bx2 = (x2 - cx1) * sx
    by2 = (y2 - cy1) * sy
    draw.rectangle([bx1, by1, bx2, by2], outline="red", width=6)
    text_thai = translate([c.cls for c in p.characters])
    label = f"{text_thai}  ({p.plate_confidence:.2f})"
    ty = max(0, by1 - 42)
    tb = draw.textbbox((bx1, ty), label, font=font)
    draw.rectangle(tb, fill="red")
    draw.text((bx1, ty), label, fill="white", font=font)
    # per-character boxes — crop uses original-image coords of the plate (cx1,cy1) as origin
    char_font = _load_font(24)
    for c in p.characters:
        # character coords are within the plate crop; map to zoom-crop coords
        ccx1 = (x1 + c.x1 - cx1) * sx
        ccy1 = (y1 + c.y1 - cy1) * sy
        ccx2 = (x1 + c.x2 - cx1) * sx
        ccy2 = (y1 + c.y2 - cy1) * sy
        draw.rectangle([ccx1, ccy1, ccx2, ccy2], outline="lime", width=2)
        glyph = to_glyph(c.cls)
        lb = draw.textbbox((ccx1, max(0, ccy1 - 26)), glyph, font=char_font)
        draw.rectangle(lb, fill="lime")
        draw.text((ccx1, max(0, ccy1 - 26)), glyph, fill="black", font=char_font)
    return crop


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
        zoom = _zoomed_crop(img, preds)
        if zoom is not None:
            zoom.save(figs_dir / f"test_{i:02d}_zoom.png")
        gallery.append(zoom if zoom is not None else annotated)
        if preds:
            row = {
                "idx": i,
                "file": p.name,
                "n_plates": len(preds),
                "best_conf": max(q.plate_confidence for q in preds),
                "codes": [q.text for q in preds],
                "thai": [translate([c.cls for c in q.characters]) for q in preds],
            }
        else:
            row = {"idx": i, "file": p.name, "n_plates": 0, "best_conf": 0.0, "codes": [], "thai": []}
        print(f"[{i:02d}] {p.name}: {row['n_plates']} plate(s), conf={row['best_conf']:.3f}, thai={row['thai']}")
        rows.append(row)

    _make_gallery(gallery, figs_dir / "test_gallery.png", cols=3)

    md = ["# Test predictions — detector validation split", ""]
    md.append("Each row is one held-out image. `thai` is the predicted plate string after")
    md.append("translating the recognizer's A## class codes via `thai_plate_ocr.char_map`.")
    md.append("")
    md.append("| idx | file | plates | conf | thai | codes |")
    md.append("|---:|---|---:|---:|---|---|")
    for r in rows:
        thai_cell = " / ".join(r.get("thai", [])) if r.get("thai") else "_(none)_"
        codes_cell = " / ".join(r.get("codes", [])) if r.get("codes") else ""
        md.append(f"| {r['idx']} | `{r['file']}` | {r['n_plates']} | {r['best_conf']:.3f} | {thai_cell} | `{codes_cell}` |")

    out_md = root / "reports/test_predictions.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    (root / "reports/test_predictions.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_md}")
    print(f"Wrote {figs_dir/'test_gallery.png'}")


if __name__ == "__main__":
    main()
