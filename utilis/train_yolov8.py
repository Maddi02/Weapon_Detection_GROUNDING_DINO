import os
import yaml
import torch
# Enable unpickling of Ultralytics model components on PyTorch 2.6+
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv
# Register safe globals for torch.load
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])

from ultralytics import YOLO


from utilis.create_data_yaml import create_data_yaml

def train_yolov8(
    train_dir: str,
    val_dir: str,
    classes_file: str,
    pretrained: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    device: str,
    output_dir: str
):
    # Load class names
    with open(classes_file, 'r') as f:
        class_names = [l.strip() for l in f if l.strip()]

    # Create data.yaml
    data_yaml = os.path.join(output_dir, 'data.yaml')
    create_data_yaml(train_dir, val_dir, class_names, data_yaml)

    # Initialize model (weights_only loading now trusts DetectionModel, Sequential, Conv)
    model = YOLO(pretrained)

    # Training
    print("Starting training...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name='yolov8_training',
        exist_ok=True
    )

    # Validation
    print("Starting validation...")
    val_results = model.val(
        data=data_yaml,
        device=device
    )

    # Print and save metrics
    metrics = val_results.metrics
    print("\nValidation Metrics:")
    print(f"Precision: {metrics.get('p', 'N/A'):.4f}")
    print(f"Recall:    {metrics.get('r', 'N/A'):.4f}")
    print(f"mAP@0.5:   {metrics.get('mAP_0.5', 'N/A'):.4f}")
    print(f"mAP@0.5:0.95: {metrics.get('mAP_0.5:0.95', 'N/A'):.4f}")

    metrics_path = os.path.join(output_dir, 'val_metrics.yaml')
    with open(metrics_path, 'w') as mf:
        yaml.dump(metrics, mf)
    print(f"Metrics saved to: {metrics_path}")