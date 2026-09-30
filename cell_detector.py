import cv2
import numpy as np
import torch
from PIL import Image

CLASS_NAMES = ['Platelets', 'RBC', 'WBC']
CLASS_NAMES_CN = ['血小板', '红细胞', '白细胞']

CLASS_COLORS = {
    'Platelets': (255, 0, 0),
    'RBC': (0, 255, 0),
    'WBC': (0, 0, 255),
}


def _imread_unicode(image_path):
    stream = open(image_path, 'rb')
    buf = np.frombuffer(stream.read(), dtype=np.uint8)
    stream.close()
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _separate_cells_by_watershed(thresh, original_img, min_area=200):
    contours = []
    orig_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    for cnt in orig_contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        if area < min_area * 3:
            contours.append(cnt)
            continue

        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() == 0:
            contours.append(cnt)
            continue

        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        _, sure_fg = cv2.threshold(dist_norm, 0.4, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)

        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel_bg, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        n_labels, markers = cv2.connectedComponents(sure_fg)
        if n_labels <= 1:
            contours.append(cnt)
            continue

        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(original_img, markers)

        split_contours = []
        for label in range(2, markers.max() + 1):
            lmask = np.uint8(markers == label) * 255
            cnts, _ = cv2.findContours(lmask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) >= min_area:
                    split_contours.append(c)

        if len(split_contours) >= 2:
            contours.extend(split_contours)
        else:
            contours.append(cnt)

    return contours


def detect_cells(image_path, model, transform, device='cpu',
                 min_area=200, max_area=50000, use_watershed=True,
                 class_names=None, class_names_cn=None, class_colors=None):
    img = _imread_unicode(image_path)
    if img is None:
        raise ValueError(f'无法读取图像: {image_path}')

    original = img.copy()
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    if use_watershed:
        contours = _separate_cells_by_watershed(thresh, img, min_area=min_area)
    else:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    if class_names is None:
        class_names = CLASS_NAMES
    if class_names_cn is None:
        class_names_cn = CLASS_NAMES_CN
    if class_colors is None:
        class_colors = CLASS_COLORS

    detections = []
    counts = {name: 0 for name in class_names}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        pad = int(max(bw, bh) * 0.15)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        cell_img = img[y1:y2, x1:x2]
        if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
            continue

        cell_pil = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
        cell_tensor = transform(cell_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(cell_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            all_scores = probs[0].cpu().numpy().tolist()

        class_name = class_names[pred_class]
        counts[class_name] += 1
        detections.append((x1, y1, x2 - x1, y2 - y1, class_name, confidence, all_scores))

    result = original.copy()
    for (x, y, bw, bh, class_name, confidence, _) in detections:
        color = class_colors[class_name]
        cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
        label = f'{class_name} {confidence:.0%}'
        cv2.putText(result, label, (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    box_h = 10 + len(class_names) * 25
    overlay = result.copy()
    cv2.rectangle(overlay, (5, 5), (200, box_h), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)

    y_offset = 28
    for class_name in class_names:
        color = class_colors[class_name]
        cn_name = class_names_cn[class_names.index(class_name)]
        text = f'{cn_name} ({class_name}): {counts[class_name]}'
        cv2.putText(result, text, (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y_offset += 25

    return result, counts, detections