import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "./data")))
    artifacts_root: Path = field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_ROOT", "./artifacts"))
    )
    detector_weights: str = os.getenv("DETECTOR_WEIGHTS", "yolov8n.pt")
    recognizer_weights: str = os.getenv("RECOGNIZER_WEIGHTS", "yolov8n.pt")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    device: str = os.getenv("DEVICE", "auto")
    seed: int = int(os.getenv("SEED", "42"))
    roboflow_api_key: str = os.getenv("ROBOFLOW_API_KEY", "")

    @property
    def detector_dataset_dir(self) -> Path:
        return self.data_root / "detector"

    @property
    def recognizer_dataset_dir(self) -> Path:
        return self.data_root / "recognizer"

    @property
    def detector_runs_dir(self) -> Path:
        return self.artifacts_root / "detector"

    @property
    def recognizer_runs_dir(self) -> Path:
        return self.artifacts_root / "recognizer"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_root,
            self.detector_dataset_dir,
            self.recognizer_dataset_dir,
            self.artifacts_root,
        ):
            p.mkdir(parents=True, exist_ok=True)
