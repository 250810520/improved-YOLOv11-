# Train two independent YOLO11 ablation variants sequentially.

import shutil
import traceback
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "lv_yolo11.yaml"
PRETRAINED = ROOT / "yolo11n.pt"
PROJECT = ROOT / "runs" / "ablation"
VARIANTS = (
    ("AIFI", ROOT / "yolo11n-sppf-aifi.yaml"),
    ("CARAFE", ROOT / "yolo11n-carafe.yaml"),
)
TRAIN_ARGS = dict(
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
)


def train_variant(name: str, model_yaml: Path) -> Path:
    run_dir = PROJECT / f"yolo11n_{name.lower()}_e150"
    source = run_dir / "weights" / "best.pt"
    target = ROOT / f"{name}-best.pt"
    if source.is_file():
        shutil.copy2(source, target)
        print(f"[{name}] existing result copied: {target}", flush=True)
        return target
    last = run_dir / "weights" / "last.pt"
    if last.is_file():
        print(f"[{name}] resuming {last}", flush=True)
        YOLO(last).train(resume=True)
    else:
        print(f"[{name}] building {model_yaml} from {PRETRAINED}", flush=True)
        YOLO(model_yaml).load(PRETRAINED).train(
            project=PROJECT, name=run_dir.name, exist_ok=False, **TRAIN_ARGS
        )
    if not source.is_file():
        raise FileNotFoundError(f"Training ended without {source}")
    shutil.copy2(source, target)
    YOLO(target).val(data=DATA, split="val", imgsz=640, batch=8, device=0, workers=0, plots=True)
    print(f"[{name}] verified: {target}", flush=True)
    return target


def main():
    for path in (DATA, PRETRAINED, *(yaml for _, yaml in VARIANTS)):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, yaml in VARIANTS:
        train_variant(name, yaml)
    print("Both ablation experiments completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

