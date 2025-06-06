# Detecting Dangerous Items in Public Space  
*A Comparison Between Zero-Shot Labeling and a Baseline YOLOv8 Pipeline*

---

## 📦 Project Overview
This repository contains two alternative training pipelines for weapon detection:

| Pipeline | Labels | Key Components |
|---------|--------|----------------|
| **Baseline YOLOv8** | Manually annotated bounding boxes from the public *Weapon Dataset for YOLOv5* | Ultralytics YOLOv8 |
| **Zero-Shot Labeling** | Bounding boxes generated on-the-fly with **Grounding DINO** prompts | Grounding DINO  →  YOLO-formatted labels  →  YOLOv8 fine-tuning |

A small **Segment Anything (SAM)** demo shows how open-set masks could further refine zero-shot boxes.

---

## 🚀 Quick Start

### 1  Install Dependencies
```bash
# Recommended: Python 3.12+ & virtual environment
pip install -r requirements.txt
```

Clone Grounding DINO
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git external/GroundingDINO
```

Smoke-Test Grounding DINO
```bash
python zero_shot/demo/pure_grounding_dino/demo.py
```

#### → bounding boxes & confidence scores should appear
Generate YOLO labels automatically
```bash
python zero_shot/utils/run_groundingdino_yolo.py
```
Train and validate both models
```bash
python train_and_validate_models.py
```
