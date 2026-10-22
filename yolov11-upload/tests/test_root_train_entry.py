from pathlib import Path


ROOT = Path(r"D:\USM\yolov11")


def test_root_training_entry_uses_approved_configuration():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    assert 'MODEL_PATH = ROOT / "yolo11n.pt"' in source
    assert 'DATA_PATH = ROOT / "lv_yolo11.yaml"' in source
    for setting in ("epochs=150", "imgsz=640", "batch=8", "device=0", "workers=0", "amp=True", "patience=0"):
        assert setting in source


def test_dataset_yaml_uses_current_project_dataset():
    source = (ROOT / "lv_yolo11.yaml").read_text(encoding="utf-8")
    assert "path: D:/USM/yolov11/datasets/lv" in source


if __name__ == "__main__":
    test_root_training_entry_uses_approved_configuration()
    test_dataset_yaml_uses_current_project_dataset()
    print("2 root training tests passed")
