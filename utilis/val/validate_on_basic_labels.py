import os

from utilis.train_yolov8 import train_yolov8
from utilis.validate_yolov8 import validate_yolov8


def validate_on_basic_labels(device: str):
    val_weights = 'runs/basic/yolov8_training/weights/best.pt'
    data_yaml = 'runs/basic/data.yaml'
    output_dir   = 'runs/val/basic'
    os.makedirs(output_dir, exist_ok=True)
    validate_yolov8(
        val_weights=val_weights,
        data_yaml=data_yaml,
        device=device,
        output_dir= output_dir,
    )