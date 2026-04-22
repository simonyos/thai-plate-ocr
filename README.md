# Thai License Plate OCR — Two-Stage YOLOv8

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonyos/thai-plate-ocr/blob/main/notebooks/thai_plate_ocr_colab.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end optical character recognition for **Thai license plates**: detect the plate
in a scene photo, crop it, then detect and spatially order the characters to produce
the plate's registration string and province.

Thai license plates are notably harder than European or US plates for off-the-shelf
OCR models because:

- Registration uses **Thai consonants** (ก, ข, ค, ...) plus **Arabic digits** (1, 2, 3, ...).
- The **province name is printed on a second line below** the registration, in Thai
  script, at a smaller font size.
- Plate **background colour signals class** (white for private cars, yellow for
  taxis / commercial, black for tuk-tuks, red for newly registered), which is not
  captured by any off-the-shelf OCR.

This project reproduces a clean two-stage pipeline and trains both stages on public
Roboflow Universe datasets.

---

## Method

Two YOLOv8 models, both fine-tuned from pre-trained COCO weights, plus a deterministic
post-processing step:

```
   ┌────────────────────┐     ┌────────────────────┐
   │  Stage 1: plate    │     │  Stage 2: per-char │
   │  detector (YOLOv8) │──▶──│  detector (YOLOv8) │──▶── order ──▶── string
   └────────────────────┘     └────────────────────┘
           │                           │
           │                           │
      1 class: plate            ~48 classes (consonants + digits + province markers)
      ~343 images                ~2,500 character-annotated images
```

The recognition stage is framed as **object detection over characters**, not as a
sequence-to-sequence OCR. This takes advantage of the character-level bounding-box
annotations available in the public dataset (which a CRNN-style reader would not use
directly) and makes the pipeline end-to-end YOLOv8 — a single inference engine for
both stages.

**Ordering characters into a string.** Once the character detector has emitted a set
of `(class, box)` tuples for a cropped plate, `thai_plate_ocr.pipeline.order_characters`
clusters them by `y`-centre with a one-dimensional gap test and sorts each cluster
left-to-right. This handles both single-line plates and the common two-line car plate
(registration on top, province below) without a separate classifier.

## Literature anchor

Character-level OCR on license plates with deep learning is most commonly framed via
CRNN / attention-based readers. The canonical reference is

> Shi, B., Bai, X., & Yao, C. (2015). An End-to-End Trainable Neural Network for
> Image-based Sequence Recognition and Its Application to Scene Text Recognition.
> [arXiv:1507.05717](https://arxiv.org/abs/1507.05717).

A CRNN reader is appropriate when you only have plate-level transcriptions as
supervision. When character-level bounding boxes are available — as they are in the
Roboflow dataset we use for stage 2 — detection over characters trained directly with
the YOLO head is simpler, faster to train, and avoids the CTC-alignment pathologies
that hurt low-data CRNN training. A full CRNN baseline is planned as a second stage-2
model for comparison once the detection-based pipeline is benchmarked.

## Datasets

Both are public Roboflow Universe datasets (CC BY 4.0). Downloaded via the Roboflow
Python SDK using `ROBOFLOW_API_KEY`; never redistributed from this repo.

| Stage | Workspace / Project | Images | Classes | Purpose |
|---|---|---:|---:|---|
| 1 | `nextra / thai-licence-plate-detect-b93xq` | 294 | 1 (`th-plate`) | Plate bounding boxes |
| 2 | `card-detector / thai-license-plate-character-detect` | 2,521 | 46 | Character bounding boxes (Thai consonants, digits, province markers) |

## Quickstart

```bash
# 1. Roboflow credentials
export ROBOFLOW_API_KEY=<paste-from-https://app.roboflow.com/settings/api>

# 2. Environment
make setup                    # uv venv + editable install

# 3. Data + training
plate download                # fetches both datasets into data/{detector,recognizer}
plate train-detector          # YOLOv8 fine-tune on plate dataset (~1h on CPU, ~10m on T4)
plate train-recognizer        # YOLOv8 fine-tune on character dataset (~3h on CPU, ~20m on T4)

# 4. Aggregate metrics
plate evaluate                # reports/summary.md + figures

# 5. End-to-end prediction
plate predict path/to/car.jpg

# 6. Inference API
make serve                    # FastAPI at http://localhost:8000/docs
```

or run the full flow on a free T4 GPU via the [Colab notebook](https://colab.research.google.com/github/simonyos/thai-plate-ocr/blob/main/notebooks/thai_plate_ocr_colab.ipynb).

## Results

> **TBD** — populated by `plate evaluate` once both stages finish training. Refer to
> `reports/summary.md` for machine-readable metrics.

| Stage | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Plate detector | — | — | — | — |
| Character detector | — | — | — | — |

A visual bar chart is saved to `reports/figures/map_by_stage.png`.

**End-to-end plate-string accuracy** — measured on held-out plate images by exact
string match against ground-truth registration — is a stronger metric than either
stage's mAP and will be added once a paired ground-truth set is available.

## Inference API

```bash
make serve
curl -F "image=@car.jpg" http://localhost:8000/predict | jq
```

Response (abridged):

```json
{
  "plates": [
    {
      "bbox_xyxy": [412, 655, 724, 742],
      "confidence": 0.93,
      "text": "กก 1234 เชียงราย",
      "text_lines": ["กก1234", "เชียงราย"],
      "characters": [
        { "cls": "ก", "conf": 0.95, "bbox_xyxy": [415, 660, 450, 710] },
        { "cls": "ก", "conf": 0.94, "bbox_xyxy": [455, 660, 490, 710] },
        { "cls": "1", "conf": 0.92, "bbox_xyxy": [500, 660, 525, 710] }
      ]
    }
  ]
}
```

The `characters` array lets any downstream consumer (web UI, automatic gate, parking
app) render per-character overlays without needing to re-run the model.

## Repository layout

```
src/thai_plate_ocr/
  config.py                 env-driven settings (paths, weights, API key, device)
  cli.py                    `plate` Typer CLI
  data/download.py          Roboflow Universe downloader
  models/detector.py        Stage-1 YOLOv8 trainer + MLflow logging
  models/recognizer.py      Stage-2 YOLOv8 trainer + MLflow logging
  pipeline.py               end-to-end detect → recognize → order → string
  evaluate.py               aggregate per-stage metrics + bar chart
  serve/api.py              FastAPI /health, /predict
tests/                      config, line-splitting, API health, CLI registration
.github/workflows/ci.yml    ruff + pytest on push and PR
Dockerfile                  slim image that serves the API
```

## Limitations and next steps

- **Two independent datasets.** Stage 1 and stage 2 are trained on different image
  distributions. End-to-end accuracy is bottlenecked by whichever stage generalises
  less well to the other's image domain. A unified dataset with both plate and
  character annotations would remove this confound.
- **No explicit province name recognition.** The stage-2 classes include a handful
  of province markers but not all 77 Thai provinces. Extending to a province-level
  classifier trained on crops of the second-line text is the natural next step.
- **No plate-colour classification.** Private vs. commercial vs. government is
  information printed by background colour alone; adding a small CNN on the plate
  crop would recover it cheaply.
- **Single seed.** Reported metrics are from one training run each.

## Citation

```bibtex
@article{shi2015crnn,
  author = {Baoguang Shi and Xiang Bai and Cong Yao},
  title  = {An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition},
  journal = {arXiv preprint arXiv:1507.05717},
  year   = {2015},
  url    = {https://arxiv.org/abs/1507.05717}
}

@misc{yosboon2026thaiplate,
  author = {Yosboon, Simon},
  title  = {Thai License Plate OCR with Two-Stage YOLOv8},
  year   = {2026},
  howpublished = {\url{https://github.com/simonyos/thai-plate-ocr}}
}
```

## License

MIT — see [LICENSE](LICENSE). Dataset licenses remain with their original authors
(CC BY 4.0 for both).
