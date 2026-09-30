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


def detect_cells(image_path, model, transform, device='cpu',
                 min_area=200, max_area=50000):
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

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    counts = {'Platelets': 0, 'RBC': 0, 'WBC': 0}

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

        class_name = CLASS_NAMES[pred_class]
        counts[class_name] += 1
        detections.append((x1, y1, x2 - x1, y2 - y1, class_name, confidence))

    result = original.copy()
    for (x, y, bw, bh, class_name, confidence) in detections:
        color = CLASS_COLORS[class_name]
        cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
        label = f'{class_name} {confidence:.0%}'
        cv2.putText(result, label, (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    overlay = result.copy()
    cv2.rectangle(overlay, (5, 5), (200, 105), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)

    y_offset = 28
    for class_name in CLASS_NAMES:
        color = CLASS_COLORS[class_name]
        cn_name = CLASS_NAMES_CN[CLASS_NAMES.index(class_name)]
        text = f'{cn_name} ({class_name}): {counts[class_name]}'
        cv2.putText(result, text, (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y_offset += 25

    return result, counts, detections