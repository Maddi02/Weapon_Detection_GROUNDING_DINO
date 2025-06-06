import os

from utilis.train_yolov8 import train_yolov8


def train_on_automated_labels(device: str):
    train_dir = '../../data/zero_shot_labels/dataset/images/train'
    val_dir = '../../data/zero_shot_labels/dataset/images/val'
    classes_file = '../../data/classes.txt'
    output_dir   = '../../runs/auto'
    os.makedirs(output_dir, exist_ok=True)
    train_yolov8(
        train_dir=train_dir,
        val_dir=val_dir,
        classes_file=classes_file,
        pretrained='yolov8n.pt',
        epochs=50,
        batch_size=16,
        img_size=640,
        device=device,
        output_dir=output_dir,
    )
