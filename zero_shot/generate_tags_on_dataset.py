import os

import torch

from zero_shot.utils.run_groundingdino_yolo import process_dataset_yolo

if __name__ == '__main__':
    base_img_dir    = '../data/origin_pictures/dataset/images'
    base_labels_dir = '../data/zero_shot_labels/dataset/labels_prompt'

    labels_file   = '../data/public_area_classes.txt'
    config_path   = '../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    weights_path  = '../GroundingDINO/weights/groundingdino_swint_ogc.pth'
    text_prompt = (
        "person | man | woman, "
        "knife | pocket knife, "
        "gun | pistol | handgun, "
        "rifle | assault rifle, "
        "shotgun, grenade, bomb, machete, "
        "sword | katana, baseball bat, crowbar"
        ", person holding knife, person holding gun, person with rifle"
    )
    box_threshold = 0.35
    text_threshold= 0.25
    iou_threshold = 0.5
    device        = 'cuda' if torch.cuda.is_available() else 'cpu'

    for split in ['train', 'val']:
        input_dir  = os.path.join(base_img_dir, split)
        output_root= os.path.join(base_labels_dir, split)
        process_dataset_yolo(
            input_dir=input_dir,
            output_root=output_root,
            labels_file=labels_file,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            iou_threshold=iou_threshold,
            device=device,
            config_path=config_path,
            weights_path=weights_path
        )
