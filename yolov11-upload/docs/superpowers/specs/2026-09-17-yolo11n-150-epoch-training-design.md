# YOLO11n 150 轮训练设计

## 目标

在 `D:\USM\yolov11` 根目录创建可直接运行的 `train.py`，加载根目录中的 `yolo11n.pt` 预训练权重，在本地 LV 数据集上训练 150 轮。

## 输入

- 预训练权重：`D:\USM\yolov11\yolo11n.pt`
- 数据集配置：`D:\USM\yolov11\lv_yolo11.yaml`
- 数据集根目录：`D:\USM\yolov11\datasets\lv`
- 训练集：1458 张图像及 1458 个标签
- 验证集：162 张图像

`lv_yolo11.yaml` 的 `path` 将修正为 `D:/USM/yolov11/datasets/lv`，避免继续读取旧工程目录。

## 训练配置

- 模型初始化：`YOLO(ROOT / "yolo11n.pt")`
- 训练轮数：`epochs=150`
- 输入尺寸：`imgsz=640`
- 批量大小：`batch=8`
- 设备：`device=0`
- Windows 数据加载：`workers=0`
- 混合精度：`amp=True`
- 输出项目：`D:\USM\yolov11\runs\detect`
- 运行名称：`lv_yolo11n_e150`
- 生成训练图表：`plots=True`

## 执行和错误处理

`train.py` 启动前检查权重、数据集 YAML 及训练/验证目录；缺失时直接报告准确路径并停止。脚本使用 `if __name__ == "__main__"` 入口，避免 Windows 多进程启动问题。训练启动后确认 CUDA、数据扫描和第一个 epoch 正常运行。

## 成功标准

1. `train.py` 可在 Python 3.9 CUDA 环境中加载本地 Ultralytics 和 `yolo11n.pt`。
2. 训练任务明确显示 `150 epochs`，并进入第一个 epoch。
3. 输出写入 `runs\detect\lv_yolo11n_e150`，最终权重位于其 `weights` 子目录。
