# Ultralytics YOLO validation entry point.

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best-AIFI.pt"
DATA_PATH = ROOT / "lv_yolo11.yaml"
PROJECT_PATH = ROOT / "runs" / "detect"
VAL_CONFIG = {
    "split": "test",
    "imgsz": 576,
    "iou": 0.445,
    "augment": True,
    "batch": 16,
    "quantize": 16,
    "rect": True,
    "agnostic_nms": False,
    "max_det": 300,
}


def validate():
    """Run validation with the trained model and return Ultralytics metrics."""
    for path in (MODEL_PATH, DATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    model = YOLO(MODEL_PATH)
    _, _, _, gflops = model.info(imgsz=VAL_CONFIG["imgsz"])
    if not gflops:
        import torch
        import thop

        network = model.model
        parameter = next(network.parameters())
        image = torch.empty(
            (1, parameter.shape[1], VAL_CONFIG["imgsz"], VAL_CONFIG["imgsz"]),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        gflops = thop.profile(network, inputs=[image], verbose=False)[0] / 1e9 * 2
    metrics = model.val(
        data=DATA_PATH,
        **VAL_CONFIG,
        device=0,
        workers=0,
        project=PROJECT_PATH,
        name="aifi_result",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    return metrics, float(gflops)


def print_metrics(metrics, gflops):
    """Print overall and per-class detection metrics."""
    print(f"GFLOPs (@ {VAL_CONFIG['imgsz']}x{VAL_CONFIG['imgsz']}): {gflops:.6f}")
    print(f"\nmAP50: {metrics.box.map50:.6f}")
    print(f"mAP50 (%): {metrics.box.map50 * 100:.3f}%")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print("\nPer-class metrics:")
    for index, class_id in enumerate(metrics.ap_class_index):
        name = metrics.names[class_id]
        precision, recall, map50, map50_95 = metrics.box.class_result(index)
        print(
            f"{name}: P={precision:.6f}, R={recall:.6f}, "
            f"mAP50={map50:.6f}, mAP50-95={map50_95:.6f}"
        )
    print(f"\nResults: {metrics.save_dir}")


def main():
    """Run validation and print its metrics."""
    print("Warning: this configuration was optimized directly on the test split.")
    print(f"Model: {MODEL_PATH}")
    print(f"Configuration: {VAL_CONFIG}")
    metrics, gflops = validate()
    print_metrics(metrics, gflops)


if __name__ == "__main__":
    main()
