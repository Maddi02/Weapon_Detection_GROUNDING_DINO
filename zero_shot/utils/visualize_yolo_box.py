import cv2


def visualize_yolo_box(img_path: str, label_path: str, output_vis_path: str):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        line = f.readline().strip().split()
    if len(line) != 5:
        cv2.imwrite(output_vis_path, img)
        return
    class_id, xc, yc, bw, bh = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
    x_center = xc * w
    y_center = yc * h
    box_w = bw * w
    box_h = bh * h
    x_min = int(x_center - box_w / 2)
    y_min = int(y_center - box_h / 2)
    x_max = int(x_center + box_w / 2)
    y_max = int(y_center + box_h / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite(output_vis_path, img)