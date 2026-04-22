"""Aggregate per-stage metrics + a comparison chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thai_plate_ocr.config import Settings


def _collect(settings: Settings) -> list[dict]:
    rows: list[dict] = []
    for name, filename in (
        ("detector", "detector_summary.json"),
        ("recognizer", "recognizer_summary.json"),
    ):
        path = settings.artifacts_root / filename
        if path.exists():
            payload = json.loads(path.read_text())
            rows.append({
                "stage": name,
                "mAP50": payload.get("mAP50"),
                "mAP50_95": payload.get("mAP50_95"),
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "train_seconds": payload.get("train_seconds"),
            })
    return rows


def _bar(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["mAP50"], width=width, label="mAP@0.5")
    ax.bar([i + width / 2 for i in x], df["mAP50_95"], width=width, label="mAP@0.5:0.95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["stage"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mAP")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(settings: Settings) -> Path:
    rows = _collect(settings)
    if not rows:
        raise FileNotFoundError(
            "No *_summary.json found under artifacts/. Run training first."
        )

    df = pd.DataFrame(rows)
    figs_dir = Path("reports/figures")
    _bar(df, figs_dir / "map_by_stage.png")

    md = ["# Results summary", "", df.to_markdown(index=False, floatfmt=".4f")]
    out_md = Path("reports/summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    return out_md
