# Ultralytics YOLO validation entry point with automatic result display.

import os
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "CARAFE-MCA-best.pt"
DATA_PATH = ROOT / "lv_yolo11.yaml"
PROJECT_PATH = ROOT / "runs" / "detect"


def validate():
    """Run validation with the trained model and return Ultralytics metrics."""
    for path in (MODEL_PATH, DATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    return YOLO(MODEL_PATH).val(
        data=DATA_PATH,
        device=0,
        workers=0,
        project=PROJECT_PATH,
        name="lv_yolo11n_val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )


def print_metrics(metrics):
    """Print overall and per-class detection metrics."""
    print(f"\nmAP50: {metrics.box.map50:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print("\nPer-class metrics:")
    for index, name in metrics.names.items():
        precision, recall, map50, map50_95 = metrics.box.class_result(index)
        print(
            f"{name}: P={precision:.6f}, R={recall:.6f}, "
            f"mAP50={map50:.6f}, mAP50-95={map50_95:.6f}"
        )
    print(f"\nResults: {metrics.save_dir}")


def show_results(result_dir):
    """Open the result folder and normalized confusion matrix on Windows."""
    result_dir = Path(result_dir).resolve()
    confusion_matrix = result_dir / "confusion_matrix_normalized.png"
    if not confusion_matrix.is_file():
        raise FileNotFoundError(f"Validation plot not found: {confusion_matrix}")
    os.startfile(str(result_dir))
    os.startfile(str(confusion_matrix))


def main():
    """Run validation, print metrics, and display generated results."""
    try:
        metrics = validate()
        print_metrics(metrics)
        show_results(metrics.save_dir)
    except Exception:
        traceback.print_exc()
        if sys.stdin and sys.stdin.isatty():
            input("\nValidation failed. Press Enter to close...")
        raise


if __name__ == "__main__":
    main()
