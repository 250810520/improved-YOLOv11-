# YOLO11n 150 Epoch Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and launch `D:\USM\yolov11\train.py` to fine-tune the local `yolo11n.pt` checkpoint for exactly 150 epochs on the local LV dataset.

**Architecture:** The root entry point validates all local inputs, imports the repository-local Ultralytics package, and calls `YOLO.train()` with the approved fixed configuration. The dataset YAML owns dataset locations and will be corrected to the current project root before training starts.

**Tech Stack:** Python 3.9, Ultralytics YOLO, PyTorch 1.13.1 CUDA 11.6, NVIDIA RTX 3060 Laptop GPU.

## Global Constraints

- Use `D:\USM\yolov11\yolo11n.pt` as pretrained weights.
- Train for `epochs=150`, `imgsz=640`, `batch=8`, `device=0`, `workers=0`, and `amp=True`.
- Save under `D:\USM\yolov11\runs\detect\lv_yolo11n_e150` with plots enabled.
- Use `D:\USM\yolov11\datasets\lv` rather than the stale dataset path in another project.
- Do not modify Ultralytics library internals.

---

### Task 1: Root training entry point and launch

**Files:**
- Create: `D:\USM\yolov11\train.py`
- Modify: `D:\USM\yolov11\lv_yolo11.yaml:1`
- Create: `D:\USM\yolov11\tests\test_root_train_entry.py`

**Interfaces:**
- Consumes: `yolo11n.pt`, `lv_yolo11.yaml`, and local image/label directories.
- Produces: `train()`, the `lv_yolo11n_e150` training run, and `best.pt`/`last.pt` under its `weights` directory.

- [ ] **Step 1: Write the failing entry-point test**

```python
from pathlib import Path


ROOT = Path(r"D:\USM\yolov11")


def test_root_training_entry_uses_approved_configuration():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    assert 'MODEL_PATH = ROOT / "yolo11n.pt"' in source
    assert 'DATA_PATH = ROOT / "lv_yolo11.yaml"' in source
    for setting in ("epochs=150", "imgsz=640", "batch=8", "device=0", "workers=0", "amp=True"):
        assert setting in source


def test_dataset_yaml_uses_current_project_dataset():
    source = (ROOT / "lv_yolo11.yaml").read_text(encoding="utf-8")
    assert "path: D:/USM/yolov11/datasets/lv" in source


if __name__ == "__main__":
    test_root_training_entry_uses_approved_configuration()
    test_dataset_yaml_uses_current_project_dataset()
    print("2 root training tests passed")
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run:

```powershell
D:\ProgramData\Anaconda3\envs\py39\python.exe D:\USM\yolov11\tests\test_root_train_entry.py
```

Expected: failure because `D:\USM\yolov11\train.py` does not yet exist.

- [ ] **Step 3: Correct the dataset root**

Replace the first YAML field with:

```yaml
path: D:/USM/yolov11/datasets/lv
```

- [ ] **Step 4: Create the training entry point**

```python
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
```

- [ ] **Step 5: Run static and entry-point tests**

Run:

```powershell
D:\ProgramData\Anaconda3\envs\py39\python.exe D:\USM\yolov11\tests\test_root_train_entry.py
D:\ProgramData\Anaconda3\envs\py39\python.exe -c "from pathlib import Path; compile(Path(r'D:\USM\yolov11\train.py').read_text(encoding='utf-8'), 'train.py', 'exec'); print('syntax=ok')"
```

Expected: `2 root training tests passed` and `syntax=ok`.

- [ ] **Step 6: Launch training and verify startup**

Run:

```powershell
D:\ProgramData\Anaconda3\envs\py39\python.exe D:\USM\yolov11\train.py
```

Expected: CUDA device 0 is selected, 1458 training and 162 validation images are scanned, the log states 150 epochs, and epoch 1 begins without an exception.

- [ ] **Step 7: Record version-control limitation**

No commit is possible because `D:\USM\yolov11` has no `.git` directory. Preserve the created files in place and report their absolute paths with the active training status.
