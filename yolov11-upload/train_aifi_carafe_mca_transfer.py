"""Train AIFI/CARAFE/MCA YOLO from a semantically remapped baseline checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_TO_TARGET_LAYERS = {16: 17, 17: 19, 19: 21, 20: 23, 22: 25, 23: 27}


def build_transfer_state(source, target):
    """Return compatible direct and semantically shifted checkpoint tensors."""
    transferred = {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}
    direct = len(transferred)
    for source_layer, target_layer in SOURCE_TO_TARGET_LAYERS.items():
        source_prefix = f"model.{source_layer}."
        target_prefix = f"model.{target_layer}."
        for source_key, value in source.items():
            if not source_key.startswith(source_prefix):
                continue
            target_key = target_prefix + source_key[len(source_prefix) :]
            if target_key in target and value.shape == target[target_key].shape:
                transferred[target_key] = value
    return transferred, {
        "direct": direct,
        "semantic": len(transferred) - direct,
        "total": len(transferred),
        "target": len(target),
    }


def train(epochs):
    """Build, transfer, and train the enhanced detector."""
    root = Path(r"D:\USM\yolo11 gaijin")
    output_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from ultralytics import YOLO

    baseline_path = root / "runs/detect/lv_yolo11n/weights/best.pt"
    model_yaml = root / "yolo11-aifi-carafe-mca.yaml"
    data_yaml = root / "lv_yolo11.yaml"
    for path in (baseline_path, model_yaml, data_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_model = YOLO(baseline_path).model
    model = YOLO(model_yaml, task="detect")
    transferred, report = build_transfer_state(source_model.state_dict(), model.model.state_dict())
    model.model.load_state_dict(transferred, strict=False)
    report["tensor_fraction"] = report["total"] / report["target"]
    report_path = output_root / "runs/aifi_carafe_mca_transfer_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Transfer report: {json.dumps(report)}")

    initialized_path = output_root / "runs/aifi_carafe_mca_semantic_transfer_init.pt"
    model.save(initialized_path)
    model = YOLO(initialized_path)

    return model.train(
        data=data_yaml,
        epochs=epochs,
        patience=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        cos_lr=True,
        amp=True,
        cache=False,
        seed=0,
        deterministic=True,
        project=output_root / "runs/detect",
        name="aifi_carafe_mca_semantic_transfer",
        exist_ok=False,
        pretrained=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    train(parser.parse_args().epochs)
