import traceback
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolo11n.pt"
DATA_PATH = ROOT / "lv_yolo11.yaml"
DATASET_PATH = ROOT / "datasets" / "lv"
PROJECT_PATH = ROOT / "runs" / "detect"


def train():
    for path in (MODEL_PATH, DATA_PATH, DATASET_PATH / "images" / "jpg-train", DATASET_PATH / "images" / "jpg-val"):
        if not path.exists():
            raise FileNotFoundError(f"Required training input not found: {path}")
    return YOLO(MODEL_PATH).train(
        data=DATA_PATH,
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        amp=True,
        patience=0,
        project=PROJECT_PATH,
        name="lv_yolo11n_e150",
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    try:
        train()
    except Exception:
        traceback.print_exc()
        raise
