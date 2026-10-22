"""Train the four remaining AIFI plus single neck-module ablation variants."""

from __future__ import annotations

import csv
import json
import shutil
import traceback
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "lv_yolo11.yaml"
PRETRAINED = ROOT / "yolo11n.pt"
PROJECT = ROOT / "runs" / "ablation"
SUMMARY = PROJECT / "aifi_single_module_remaining4_summary.csv"
VARIANTS = (
    ("CARAFE2", ROOT / "yolo11-aifi-carafe2.yaml", {}),
    ("MCA1", ROOT / "yolo11-aifi-mca1.yaml", {16: 17, 17: 18, 19: 20, 20: 21, 22: 23, 23: 24}),
    ("MCA2", ROOT / "yolo11-aifi-mca2.yaml", {17: 18, 19: 20, 20: 21, 22: 23, 23: 24}),
    ("MCA3", ROOT / "yolo11-aifi-mca3.yaml", {20: 21, 22: 23, 23: 24}),
)
TRAIN_ARGS = dict(
    data=DATA,
    epochs=150,
    imgsz=640,
    device=0,
    workers=0,
    amp=True,
    patience=0,
    seed=0,
    deterministic=True,
    plots=True,
)
BATCH_BY_VARIANT = {"CARAFE2": 4, "MCA3": 2}
BATCH_REPORT_BY_VARIANT = {"MCA3": "8 through epoch 132; 2 from epoch 133"}
SUMMARY_FIELDS = (
    "variant",
    "train_batch",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "parameters",
    "GFLOPs",
    "weight",
)


def transfer_state(source, target, layer_map):
    """Return compatible direct weights plus weights remapped after an inserted MCA layer."""
    transferred = {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}
    direct = len(transferred)
    for source_layer, target_layer in layer_map.items():
        source_prefix = f"model.{source_layer}."
        target_prefix = f"model.{target_layer}."
        for source_key, value in source.items():
            if source_key.startswith(source_prefix):
                target_key = target_prefix + source_key[len(source_prefix) :]
                if target_key in target and value.shape == target[target_key].shape:
                    transferred[target_key] = value
    return transferred, {
        "direct": direct,
        "semantic": len(transferred) - direct,
        "total": len(transferred),
        "target": len(target),
    }


def initialized_model(model_yaml: Path, layer_map: dict[int, int], report_path: Path) -> YOLO:
    """Build one variant and seed all semantically corresponding layers from YOLO11n."""
    source = YOLO(PRETRAINED).model
    model = YOLO(model_yaml, task="detect").load(PRETRAINED)
    transferred, report = transfer_state(source.state_dict(), model.model.state_dict(), layer_map)
    model.model.load_state_dict(transferred, strict=False)
    report["tensor_fraction"] = report["total"] / report["target"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Transfer report: {json.dumps(report)}", flush=True)
    return model


def metric_value(metrics, key: str) -> float:
    """Read one scalar validation metric."""
    return float(metrics.results_dict[key])


def write_summary(row: dict) -> None:
    """Update the restart-safe four-variant summary."""
    rows = {}
    if SUMMARY.is_file():
        with SUMMARY.open(newline="", encoding="utf-8-sig") as file:
            rows = {item["variant"]: item for item in csv.DictReader(file)}
    rows[row["variant"]] = row
    order = {name: index for index, (name, _, _) in enumerate(VARIANTS)}
    temporary = SUMMARY.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda item: order[item["variant"]]))
    temporary.replace(SUMMARY)


def has_final_summary(name: str, weight: Path) -> bool:
    """Skip validation when a full training run already has its final result."""
    if not weight.is_file() or not SUMMARY.is_file():
        return False
    with SUMMARY.open(newline="", encoding="utf-8-sig") as file:
        return any(
            row.get("variant") == name and row.get("weight") == str(weight)
            for row in csv.DictReader(file)
        )


def completed_epochs(run_dir: Path) -> int:
    """Return the latest epoch with both metrics and a saved checkpoint."""
    results = run_dir / "results.csv"
    if not results.is_file():
        return 0
    with results.open(newline="", encoding="utf-8-sig") as file:
        epoch = max((int(float(row["epoch"])) for row in csv.DictReader(file)), default=0)
    last = run_dir / "weights" / "last.pt"
    if epoch and (not last.is_file() or last.stat().st_mtime_ns < results.stat().st_mtime_ns):
        return epoch - 1  # Ultralytics writes results.csv before last.pt each epoch.
    return epoch


def train_variant(name: str, model_yaml: Path, layer_map: dict[int, int]) -> Path:
    """Train or resume one single-module variant, then run a fixed validation pass."""
    run_dir = PROJECT / f"yolo11n_aifi_{name.lower()}_e150"
    source = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    target = ROOT / f"AIFI-{name}-best.pt"
    train_batch = BATCH_BY_VARIANT.get(name, 8)
    run_dir.mkdir(parents=True, exist_ok=True)
    finished_epochs = completed_epochs(run_dir)

    if finished_epochs >= TRAIN_ARGS["epochs"] and source.is_file() and has_final_summary(name, target):
        print(f"[{name}] final result already recorded in {SUMMARY}", flush=True)
        return target
    if finished_epochs >= TRAIN_ARGS["epochs"] and source.is_file():
        print(f"[{name}] using completed {finished_epochs}-epoch checkpoint {source}", flush=True)
    elif last.is_file():
        print(f"[{name}] resuming after epoch {finished_epochs} from {last} with batch={train_batch}", flush=True)
        resume_args = {"batch": train_batch}
        if name == "MCA3":
            resume_args["plots"] = False  # reduce host-memory pressure without changing the training data
        YOLO(last).train(resume=True, **resume_args)
    else:
        print(f"[{name}] building {model_yaml.name} from {PRETRAINED.name}", flush=True)
        model = initialized_model(model_yaml, layer_map, run_dir / "transfer_report.json")
        model.train(project=PROJECT, name=run_dir.name, exist_ok=True, batch=train_batch, **TRAIN_ARGS)

    finished_epochs = completed_epochs(run_dir)
    if finished_epochs < TRAIN_ARGS["epochs"] or not source.is_file():
        raise RuntimeError(f"[{name}] stopped at epoch {finished_epochs}; expected {TRAIN_ARGS['epochs']} and {source}")
    shutil.copy2(source, target)

    model = YOLO(target)
    _, parameters, _, gflops = model.info(imgsz=640)
    metrics = model.val(
        data=DATA,
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=run_dir,
        name="final_val",
        exist_ok=True,
    )
    write_summary(
        {
            "variant": name,
            "train_batch": BATCH_REPORT_BY_VARIANT.get(name, train_batch),
            "precision": metric_value(metrics, "metrics/precision(B)"),
            "recall": metric_value(metrics, "metrics/recall(B)"),
            "mAP50": metric_value(metrics, "metrics/mAP50(B)"),
            "mAP50-95": metric_value(metrics, "metrics/mAP50-95(B)"),
            "preprocess_ms": float(metrics.speed["preprocess"]),
            "inference_ms": float(metrics.speed["inference"]),
            "postprocess_ms": float(metrics.speed["postprocess"]),
            "parameters": parameters,
            "GFLOPs": gflops if gflops > 0 else "",
            "weight": str(target),
        }
    )
    print(f"[{name}] verified weight: {target}", flush=True)
    return target


def main() -> None:
    """Run the four remaining single-position experiments sequentially on GPU 0."""
    for path in (DATA, PRETRAINED, *(model_yaml for _, model_yaml, _ in VARIANTS)):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, model_yaml, layer_map in VARIANTS:
        train_variant(name, model_yaml, layer_map)
    print(f"All four experiments completed. Summary: {SUMMARY}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
