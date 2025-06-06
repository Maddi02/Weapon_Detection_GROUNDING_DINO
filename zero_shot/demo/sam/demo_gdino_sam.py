from __future__ import annotations
import cv2, torch, numpy as np
from pathlib import Path
from torchvision.ops import nms
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

# ---------------- feste Parameter -----------------
GDINO_CFG     = "../../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS = "../../../GroundingDINO/weights/groundingdino_swint_ogc.pth"
SAM_CKPT      = "sam_vit_h_4b8939.pth"
IMAGE_PATH    = "../../../data/origin_pictures/dataset/images/train/1.jpg"
PROMPT        = "gun . knife . firearm . pistol . revolver ."
BOX_THR, TXT_THR, IOU_NMS = 0.45, 0.25, 0.45
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR       = Path("demo_out")
TEXT_THR = 0.40
SAM_W = "sam_vit_h_4b8939.pth"
def main() -> None:

    # ---------- Grounding-DINO --------------------------------------
    dino = load_model(GDINO_CFG, GDINO_WEIGHTS).to(DEVICE)
    img_bgr, img_tsr = load_image(IMAGE_PATH)
    boxes, scores, phrases = predict(dino, img_tsr, PROMPT,
                                     BOX_THR, TEXT_THR, DEVICE)
    if not len(boxes):
        raise SystemExit("Keine Box detektiert.")

    # NMS + Box mit höchstem Score auswählen
    keep  = nms(boxes, scores, IOU_NMS)
    best  = torch.argmax(scores[keep])
    box_n = boxes[keep][best]         # normiert [0,1]
    phrase= phrases[keep[best]]
    H, W  = img_bgr.shape[:2]

    # Normierte Box → Pixel
    box_px = box_n.clone()
    box_px[[0,2]] *= W
    box_px[[1,3]] *= H
    print("Pixel-Box:", box_px.tolist(), "|", phrase)

    # ---------- SAM -------------------------------------------------
    sam   = sam_model_registry["vit_h"](checkpoint=SAM_W).to(DEVICE)
    pred  = SamPredictor(sam)
    pred.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Box in SAM-Koordinaten transformieren
    sam_box = pred.transform.apply_boxes_torch(
                 box_px.unsqueeze(0), (H, W)
              ).squeeze(0).cpu().numpy().astype(np.float32)

    # optionaler Punktprompt (Box-Mitte) erhöht Erfolgswahrscheinlichkeit
    cx, cy = (box_px[0]+box_px[2])/2, (box_px[1]+box_px[3])/2
    pt_c   = np.array([[cx, cy]], np.float32)
    pt_l   = np.array([1], np.int32)

    mask, _, _ = pred.predict(box=sam_box,
                              point_coords=pt_c,
                              point_labels=pt_l,
                              multimask_output=False)
    mask = mask.squeeze()
    cv2.imwrite(str(OUT_DIR/"mask.png"), mask.astype(np.uint8)*255)

    # ---------- Overlay zur Kontrolle --------------------------------
    overlay = img_bgr.copy()
    cnts,_ = cv2.findContours(mask.astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0,255,0), 2)
    x1,y1,x2,y2 = map(int, box_px)
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 1)
    cv2.imwrite(str(OUT_DIR/"overlay.png"), overlay)
    print("Maske & Overlay →", OUT_DIR.resolve())

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()