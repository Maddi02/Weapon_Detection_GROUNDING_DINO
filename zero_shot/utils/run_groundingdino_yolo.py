from __future__ import annotations

import os
import torch
from torchvision.ops import nms
import cv2

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from zero_shot.utils.visualize_yolo_box import visualize_yolo_box


def save_boxes_as_yolo(
    boxes: torch.Tensor,
    phrases: list[str],
    out_path: str,
    class_map: dict[str, int] | None = None
):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = boxes.size(0)

    if class_map is None:
        class_ids = torch.zeros(N, dtype=torch.int)
    else:
        class_ids = torch.tensor([class_map.get(p, 0) for p in phrases], dtype=torch.int)

    yolo = torch.cat([class_ids.unsqueeze(1).float(), boxes], dim=1)

    with open(out_path, "w") as f:
        for row in yolo.tolist():
            cls_id = int(row[0])
            coords = [f"{v:.6f}" for v in row[1:]]
            f.write(f"{cls_id} {' '.join(coords)}\n")


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> list[int]:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    keep = nms(xyxy, scores, iou_threshold)
    return keep.tolist()


def process_dataset_yolo(
    input_dir: str,
    output_root: str,
    labels_file: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    iou_threshold: float,
    device: str,
    config_path: str,
    weights_path: str
):

    model = load_model(config_path, weights_path)
    model.to(device)

    with open(labels_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    class_map = {cls_name: idx for idx, cls_name in enumerate(classes)}

    image_patterns = ['.jpg', '.jpeg', '.png']
    image_paths: list[str] = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in image_patterns):
                image_paths.append(os.path.join(root, fname))
    if not image_paths:
        raise RuntimeError(f"Keine Bilder gefunden in {input_dir}")
    image_paths.sort()

    for img_path in image_paths:
        _, image = load_image(img_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )

        if iou_threshold > 0:
            keep = apply_nms(boxes, logits, iou_threshold)
            boxes  = boxes[keep]
            logits = logits[keep]
            phrases = [phrases[i] for i in keep]

        rel_path = os.path.relpath(img_path, input_dir)
        base     = os.path.splitext(rel_path)[0]
        out_dir  = os.path.join(output_root, os.path.dirname(rel_path))
        os.makedirs(out_dir, exist_ok=True)
        out_txt  = os.path.join(out_dir, f"{os.path.basename(base)}.txt")
        save_boxes_as_yolo(boxes, phrases, out_txt, class_map)
        print(f"[YOLO] Gespeichert: {out_txt}")

        vis_dir = os.path.join(output_root, 'vis', os.path.dirname(rel_path))
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{os.path.basename(base)}.jpg")
        visualize_yolo_box(img_path, out_txt, vis_path)
        print(f"[VIS]  Gespeichert: {vis_path}")
