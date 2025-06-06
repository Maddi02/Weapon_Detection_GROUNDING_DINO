import cv2
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import torch

def main():
    config_path = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "../GroundingDINO/weights/groundingdino_swint_ogc.pth"
    image_path = "../data/origin_pictures/dataset/images/train/1.jpg"
    text_prompt = "gun . knife ."
    box_threshold = 0.35
    text_threshold = 0.25


    phrase2id = {
        "gun":   0,
        "knife": 1,

    }


    model = load_model(config_path, weights_path)


    image_source, image = load_image(image_path)


    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu"
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    output_path = "demo/annotated_image.jpg"
    cv2.imwrite(output_path, annotated_frame)

    if hasattr(image_source, "shape"):
        img_h, img_w = image_source.shape[:2]
    else:
        img_w, img_h = image_source.size

    # Konsolen-Ausgaben
    total_boxes = len(boxes)
    print(f"Anzahl erkannter Boxen: {total_boxes}\n")

    for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases), start=1):
        # Score extrahieren
        try:
            score = logit.sigmoid().item() if isinstance(logit, torch.Tensor) else float(logit)
        except Exception:
            score = float(logit)

        # Bounding-Box-Koordinaten
        if hasattr(box, "cpu"):
            box = box.cpu().numpy()
        x1, y1, x2, y2 = box

        # YOLO-Formeln
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        box_w    = x2 - x1
        box_h    = y2 - y1

        # Normalisierung auf [0,1]
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        width_norm    = box_w    / img_w
        height_norm   = box_h    / img_h

        # Klassen-ID
        class_id = phrase2id.get(phrase, -1)  # falls unmapped: -1

        # Ausgabe
        print(f"Box {idx}: {phrase}, Wahrscheinlichkeit: {score:.2f}")
        if class_id >= 0:
            print(
                f"  YOLO → {class_id} "
                f"{x_center_norm:.6f} {y_center_norm:.6f} "
                f"{width_norm:.6f} {height_norm:.6f}\n"
            )
        else:
            print("  [Phrase nicht im phrase2id-Mapping, YOLO-Ausgabe übersprungen]\n")

if __name__ == "__main__":
    main()
