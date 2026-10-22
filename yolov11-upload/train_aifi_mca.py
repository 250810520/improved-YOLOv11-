# Train the combined YOLO11 AIFI + four-Neck-MCA experiment.

import shutil
import traceback
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "yolo11n-aifi-mca.yaml"
PRETRAINED = ROOT / "yolo11n.pt"
DATA = ROOT / "lv_yolo11.yaml"
PROJECT = ROOT / "runs" / "ablation"
RUN = PROJECT / "yolo11n_aifi_mca_e150"
TARGET = ROOT / "AIFI-MCA-best.pt"


def train():
    for path in (MODEL, PRETRAINED, DATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = RUN / "weights" / "best.pt"
    last = RUN / "weights" / "last.pt"
    if source.is_file():
        shutil.copy2(source, TARGET)
        return TARGET
    if last.is_file():
        YOLO(last).train(resume=True)
    else:
        YOLO(MODEL).load(PRETRAINED).train(
            data=DATA,
            epochs=150,
            imgsz=640,
            batch=8,
            device=0,
            workers=0,
            amp=True,
            patience=0,
            seed=0,
            deterministic=True,
            plots=True,
            project=PROJECT,
            name=RUN.name,
            exist_ok=False,
        )
    if not source.is_file():
        raise FileNotFoundError(f"Training ended without {source}")
    shutil.copy2(source, TARGET)
    YOLO(TARGET).val(data=DATA, split="val", imgsz=640, batch=8, device=0, workers=0, plots=True)
    return TARGET


if __name__ == "__main__":
    try:
        print(f"Verified weight: {train()}", flush=True)
    except Exception:
        traceback.print_exc()
        raise

