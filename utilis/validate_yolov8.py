import os

import yaml
from ultralytics import YOLO

def validate_yolov8(
    val_weights: str,
    data_yaml: str,
    device: str,
    output_dir: str
):
    model = YOLO(val_weights)
    print("Starting validation...")
    val_results = model.val(
        data=data_yaml,
        device=device,
        save = True,
        project = output_dir,
        name = 'val',
        exist_ok = True
    )
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
