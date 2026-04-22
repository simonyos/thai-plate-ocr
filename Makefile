.PHONY: setup data train-detector train-recognizer eval serve test lint fmt docker clean

PY ?= python
UV ?= uv

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev]"

data:
	$(PY) -m thai_plate_ocr.cli download

train-detector:
	$(PY) -m thai_plate_ocr.cli train-detector

train-recognizer:
	$(PY) -m thai_plate_ocr.cli train-recognizer

eval:
	$(PY) -m thai_plate_ocr.cli evaluate

serve:
	uvicorn thai_plate_ocr.serve.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

docker:
	docker build -t thai-plate-ocr:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov mlruns runs artifacts
