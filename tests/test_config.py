from pathlib import Path

from thai_plate_ocr.config import Settings


def test_derived_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "art"))
    s = Settings()
    assert s.detector_dataset_dir == Path(tmp_path / "data" / "detector")
    assert s.recognizer_dataset_dir == Path(tmp_path / "data" / "recognizer")
    assert s.detector_runs_dir == Path(tmp_path / "art" / "detector")


def test_ensure_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "art"))
    s = Settings()
    s.ensure_dirs()
    assert s.detector_dataset_dir.is_dir()
    assert s.recognizer_dataset_dir.is_dir()
    assert s.artifacts_root.is_dir()
