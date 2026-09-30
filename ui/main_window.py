import os
import sys
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TORCH_AVAILABLE = False
_torch = None
_transforms = None
_ResNet = None

try:
    import torch
    _torch = torch
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.environ['PATH'] = _torch_lib + ';' + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_torch_lib)
    for _dll in ['c10.dll', 'torch_global_deps.dll']:
        try:
            ctypes.CDLL(os.path.join(_torch_lib, _dll))
        except:
            pass
    del _torch_lib, _dll
    import torchvision.transforms as _transforms
    from model import ResNet as _ResNet
    import torchvision.datasets as _torchvision_datasets
    from torch.utils.data import DataLoader
    import cv2
    TORCH_AVAILABLE = True
except Exception:
    pass

import PyQt5
_qt5_root = os.path.dirname(PyQt5.__file__)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(_qt5_root, 'Qt5', 'plugins', 'platforms')

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QFileDialog, QTextEdit, QGroupBox,
    QGridLayout, QFrame, QSplitter, QProgressBar, QSpinBox,
    QDoubleSpinBox, QComboBox, QLineEdit, QMessageBox, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

_CN_FONTS = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
for _fn in _CN_FONTS:
    try:
        import matplotlib.font_manager as _fm
        _found = [f for f in _fm.fontManager.ttflist if f.name == _fn]
        if _found:
            matplotlib.rcParams['font.sans-serif'] = [_fn, 'DejaVu Sans'] + matplotlib.rcParams.get('font.sans-serif', [])
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
    except Exception:
        continue
del _CN_FONTS, _fn, _found, _fm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

from option import get_args

opt = get_args()
CLASS_NAMES = ['Platelets', 'RBC', 'WBC']
CLASS_NAMES_CN = ['血小板', '红细胞', '白细胞']


def _check_torch():
    global TORCH_AVAILABLE
    if TORCH_AVAILABLE and _torch and _transforms and _ResNet:
        return True, _torch, _transforms, _ResNet
    QMessageBox.warning(
        None, 'PyTorch 不可用',
        'PyTorch 未能成功加载，模型相关功能不可用。\n\n'
        '请检查 Python 与 PyTorch 版本的兼容性，'
        '或尝试使用 Python 3.11/3.12。'
    )
    return False, None, None, None


class PredictThread(QThread):
    result_signal = pyqtSignal(int, float, np.ndarray)

    def __init__(self, model, image_path, transform, loadsize):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.transform = transform
        self.loadsize = loadsize

    def run(self):
        try:
            image = Image.open(self.image_path).convert('RGB')
            original = np.array(image)
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

            img_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            self.result_signal.emit(pred_class, confidence, original)
        except Exception as e:
            self.result_signal.emit(-1, 0.0, None)


class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, args_dict):
        super().__init__()
        self.args_dict = args_dict

    def run(self):
        try:
            nn = torch.nn
            optim = torch.optim
            from getdata import MyData
            import getdata
            device = torch.device(self.args_dict.get('device', 'cpu'))
            self.log_signal.emit(f'设备: {device}')

            self.log_signal.emit('正在创建模型...')
            model = _ResNet(num_classes=3, pretrained=True).to(device)
            self.log_signal.emit('模型创建完成 (ResNet50)')

            checkpoints_dir = self.args_dict.get('checkpoints', './checkpoints/')
            log_dir = self.args_dict.get('log_dir', './log_dir')
            logging_txt = self.args_dict.get('logging_txt', './log_dir/logging.txt')

            os.makedirs(checkpoints_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            if 'dataset_train' in self.args_dict:
                getdata.opt.dataset_train = self.args_dict['dataset_train']
            if 'dataset_val' in self.args_dict:
                getdata.opt.dataset_val = self.args_dict['dataset_val']
            if 'dataset_test' in self.args_dict:
                getdata.opt.dataset_test = self.args_dict['dataset_test']
            if 'batch_size' in self.args_dict:
                getdata.opt.batch_size = self.args_dict['batch_size']
            if 'loadsize' in self.args_dict:
                getdata.opt.loadsize = self.args_dict['loadsize']

            self.log_signal.emit('正在加载数据集...')
            dataloaders = MyData()
            train_loader = dataloaders['train']
            val_loader = dataloaders['val']
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            classes = train_loader.dataset.classes
            self.log_signal.emit(f'训练集: {train_size} 张 | 验证集: {val_size} 张')
            self.log_signal.emit(f'类别: {classes}')

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.args_dict.get('lr', 0.001))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            epochs = self.args_dict.get('epochs', 3)
            best_acc = 0.0

            self.log_signal.emit(f'开始训练，共 {epochs} 轮，每轮 {train_size} 张图像...')

            for epoch in range(epochs):
                self.log_signal.emit(f'--- 第 {epoch + 1}/{epochs} 轮开始 ---')
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                train_loss_avg = train_loss / len(train_loader)
                train_acc = train_correct / train_total

                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss_avg = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                scheduler.step()

                self.log_signal.emit(
                    f'Epoch {epoch + 1}/{epochs} | '
                    f'Train Loss: {train_loss_avg:.4f} Acc: {train_acc:.4f} | '
                    f'Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.4f}'
                )

                progress = int((epoch + 1) / epochs * 100)
                self.progress_signal.emit(progress)

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), checkpoints_dir + 'best.pth')
                    self.log_signal.emit(f'>>> 最佳模型已保存 (acc: {best_acc:.4f})')

                if (epoch + 1) % 5 == 0:
                    torch.save(model.state_dict(), checkpoints_dir + f'epoch_{epoch + 1}.pth')

            self.finished_signal.emit(True, f'训练完成！最佳验证精度: {best_acc:.4f}')
        except Exception as e:
            self.finished_signal.emit(False, f'训练失败: {str(e)}')


class ConvertThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, pth_path, onnx_path, loadsize):
        super().__init__()
        self.pth_path = pth_path
        self.onnx_path = onnx_path
        self.loadsize = loadsize

    def run(self):
        try:
            self.log_signal.emit(f'加载模型: {self.pth_path}')
            model = _ResNet(num_classes=3, pretrained=False)
            ckpt = torch.load(self.pth_path, map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
            model.eval()

            self.log_signal.emit('开始转换...')
            dummy_input = torch.randn(1, 3, self.loadsize, self.loadsize)

            torch.onnx.export(
                model, dummy_input, self.onnx_path,
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                opset_version=11,
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

            self.finished_signal.emit(True, f'转换成功！\n{self.onnx_path}')
        except Exception as e:
            self.finished_signal.emit(False, f'转换失败: {str(e)}')


class DetectThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, object, object)

    def __init__(self, image_path, model, transform, loadsize, min_area, max_area):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.transform = transform
        self.loadsize = loadsize
        self.min_area = min_area
        self.max_area = max_area

    def run(self):
        from cell_detector import detect_cells
        try:
            device = 'cpu'
            result_img, counts, detections = detect_cells(
                self.image_path, self.model, self.transform,
                device=device, min_area=self.min_area, max_area=self.max_area
            )
            total = sum(counts.values())
            self.log_signal.emit(f'检测完成: 共发现 {total} 个细胞')
            self.log_signal.emit(f'  白细胞(WBC): {counts["WBC"]}')
            self.log_signal.emit(f'  红细胞(RBC): {counts["RBC"]}')
            self.log_signal.emit(f'  血小板(Platelets): {counts["Platelets"]}')
            self.finished_signal.emit(True, '', result_img, counts)
        except Exception as e:
            self.finished_signal.emit(False, f'检测失败: {str(e)}', None, None)


class ValidateThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str, object, object, object, object)

    def __init__(self, model_path, val_data_path, loadsize, num_classes=3):
        super().__init__()
        self.model_path = model_path
        self.val_data_path = val_data_path
        self.loadsize = loadsize
        self.num_classes = num_classes

    def run(self):
        try:
            self.log_signal.emit('正在加载模型...')
            model = _ResNet(num_classes=self.num_classes, pretrained=False)
            ckpt = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(ckpt, strict=False)
            model.eval()

            transform_test = _transforms.Compose([
                _transforms.Resize((self.loadsize, self.loadsize)),
                _transforms.ToTensor(),
                _transforms.Normalize([0.6786, 0.6413, 0.6605],
                                      [0.2599, 0.2595, 0.2569])
            ])

            self.log_signal.emit(f'正在加载验证集: {self.val_data_path}')
            val_dataset = _torchvision_datasets.ImageFolder(
                root=self.val_data_path, transform=transform_test
            )
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

            class_names = val_dataset.classes
            self.log_signal.emit(f'类别: {class_names}')
            self.log_signal.emit(f'样本总数: {len(val_dataset)}')

            self.progress_signal.emit(10)

            self.log_signal.emit('正在进行推理...')
            y_true = []
            y_pred = []
            y_scores = []

            total_batches = len(val_loader)
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    outputs = model(images)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    y_scores.append(probs.cpu().numpy())
                    pct = 10 + int(60 * (batch_idx + 1) / total_batches)
                    self.progress_signal.emit(pct)

            y_scores = np.concatenate(y_scores, axis=0)

            self.log_signal.emit('正在计算评估指标...')
            n_classes = len(class_names)

            accuracy = accuracy_score(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision_per = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
            report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

            metrics_text = (
                f'[accuracy: {accuracy:.4f}]  '
                f'[precision: {precision_macro:.4f}]  '
                f'[recall: {recall_macro:.4f}]  '
                f'[f1: {f1_macro:.4f}]\n\n{report}'
            )
            self.log_signal.emit('\n' + metrics_text)

            per_class_metrics = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_per': list(precision_per),
                'recall_per': list(recall_per),
                'f1_per': list(f1_per),
                'class_names': class_names,
            }

            self.progress_signal.emit(75)

            self.log_signal.emit('正在生成混淆矩阵...')
            conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

            self.log_signal.emit('正在生成ROC曲线...')
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_true_bin.ravel(), y_scores.ravel()
            )
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            self.progress_signal.emit(90)

            roc_data = {
                'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
                'n_classes': n_classes, 'class_names': class_names
            }

            self.log_signal.emit('验证完成！')
            self.progress_signal.emit(100)

            self.finished_signal.emit(True, '', conf_matrix, roc_data, class_names, per_class_metrics)

        except Exception as e:
            self.finished_signal.emit(False, f'验证失败: {str(e)}', None, None, None, None)


class MultiCellValThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str, object, object, object, object)

    def __init__(self, val_dir, model, transform, loadsize, min_area, max_area, iou_thresh=0.3):
        super().__init__()
        self.val_dir = val_dir
        self.model = model
        self.transform = transform
        self.loadsize = loadsize
        self.min_area = min_area
        self.max_area = max_area
        self.iou_thresh = iou_thresh

    @staticmethod
    def _compute_iou(box_a, box_b):
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / float(area_a + area_b - inter)

    def _detect_with_scores(self, image_path):
        import cv2 as cv

        stream = open(image_path, 'rb')
        buf = np.frombuffer(stream.read(), dtype=np.uint8)
        stream.close()
        img = cv.imdecode(buf, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f'无法读取图像: {image_path}')

        h, w = img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blurred, 0, 255,
                                  cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_SIMPLE)

        detections = []
        counts = {'Platelets': 0, 'RBC': 0, 'WBC': 0}

        for contour in contours:
            area = cv.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, bw, bh = cv.boundingRect(contour)
            pad = int(max(bw, bh) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            cell_img = img[y1:y2, x1:x2]
            if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                continue
            cell_pil = Image.fromarray(cv.cvtColor(cell_img, cv.COLOR_BGR2RGB))
            cell_tensor = self.transform(cell_pil).unsqueeze(0)
            with torch.no_grad():
                output = self.model(cell_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                all_scores = probs[0].cpu().numpy()
            class_name = CLASS_NAMES[pred_class]
            counts[class_name] += 1
            detections.append({
                'box': [x1, y1, x2, y2],
                'class': class_name,
                'class_id': pred_class,
                'confidence': confidence,
                'scores': all_scores,
            })
        return counts, detections

    def run(self):
        import glob
        import xml.etree.ElementTree as ET

        try:
            image_files = glob.glob(os.path.join(self.val_dir, '*.jpg'))
            if not image_files:
                image_files = glob.glob(os.path.join(self.val_dir, '*.png'))
            if not image_files:
                image_files = glob.glob(os.path.join(self.val_dir, '*.jpeg'))
            if not image_files:
                self.finished_signal.emit(False, '验证目录中没有找到图像文件', None, None, None, None)
                return

            self.log_signal.emit(f'找到 {len(image_files)} 张多细胞验证图像')
            self.log_signal.emit(f'IoU 匹配阈值: {self.iou_thresh}')
            self.progress_signal.emit(0)

            class_names = CLASS_NAMES
            n_classes = len(class_names)
            all_y_true = []
            all_y_pred = []
            all_y_scores = []

            total_gt = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
            total_pred = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
            total_tp = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
            image_results = []

            for idx, img_path in enumerate(image_files):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                xml_path = os.path.join(self.val_dir, base_name + '.xml')

                gt_boxes = []
                gt_counts = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
                if os.path.exists(xml_path):
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        for obj in root.findall('object'):
                            name_el = obj.find('name')
                            bndbox = obj.find('bndbox')
                            if name_el is None or bndbox is None:
                                continue
                            cls_name = name_el.text
                            if cls_name not in class_names:
                                continue
                            xmin = int(float(bndbox.find('xmin').text))
                            ymin = int(float(bndbox.find('ymin').text))
                            xmax = int(float(bndbox.find('xmax').text))
                            ymax = int(float(bndbox.find('ymax').text))
                            gt_boxes.append([xmin, ymin, xmax, ymax, cls_name,
                                             class_names.index(cls_name)])
                            gt_counts[cls_name] += 1
                    except Exception as e:
                        self.log_signal.emit(f'  [警告] 解析 {os.path.basename(xml_path)} 失败: {e}')

                for k in gt_counts:
                    total_gt[k] += gt_counts[k]

                try:
                    pred_counts, detections = self._detect_with_scores(img_path)
                except Exception as e:
                    self.log_signal.emit(f'  [警告] {os.path.basename(img_path)} 检测失败: {e}')
                    pct = int((idx + 1) / len(image_files) * 100)
                    self.progress_signal.emit(pct)
                    continue

                for k in pred_counts:
                    total_pred[k] += pred_counts[k]

                matched_gt = set()
                matched_pred = set()

                if gt_boxes and detections:
                    iou_matrix = np.zeros((len(gt_boxes), len(detections)))
                    for gi, gt in enumerate(gt_boxes):
                        for di, det in enumerate(detections):
                            iou_matrix[gi, di] = self._compute_iou(
                                gt[:4], det['box']
                            )

                    while True:
                        max_iou = iou_matrix.max()
                        if max_iou < self.iou_thresh:
                            break
                        gi, di = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                        gt_cls = gt_boxes[gi][4]
                        gt_cls_id = gt_boxes[gi][5]
                        pred_cls = detections[di]['class']
                        pred_cls_id = detections[di]['class_id']
                        all_y_true.append(gt_cls_id)
                        all_y_pred.append(pred_cls_id)
                        all_y_scores.append(detections[di]['scores'])
                        if gt_cls == pred_cls:
                            total_tp[gt_cls] += 1
                        matched_gt.add(gi)
                        matched_pred.add(di)
                        iou_matrix[gi, :] = 0
                        iou_matrix[:, di] = 0

                fp_per_class = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
                fn_per_class = {'WBC': 0, 'RBC': 0, 'Platelets': 0}
                for gi in range(len(gt_boxes)):
                    if gi not in matched_gt:
                        fn_per_class[gt_boxes[gi][4]] += 1
                for di in range(len(detections)):
                    if di not in matched_pred:
                        fp_per_class[detections[di]['class']] += 1

                image_results.append({
                    'name': os.path.basename(img_path),
                    'gt': dict(gt_counts),
                    'pred': dict(pred_counts),
                    'tp': dict(total_tp),
                    'fp': fp_per_class,
                    'fn': fn_per_class,
                    'matched': len(matched_gt),
                    'has_gt': os.path.exists(xml_path),
                })

                pct = int((idx + 1) / len(image_files) * 100)
                self.progress_signal.emit(pct)

            if len(all_y_true) == 0:
                self.log_signal.emit('\n⚠ 没有可匹配的细胞对，无法生成评估指标')
                self.progress_signal.emit(100)
                self.finished_signal.emit(True, '', None, None, None, None)
                return

            y_true = np.array(all_y_true)
            y_pred = np.array(all_y_pred)
            y_scores = np.array(all_y_scores)

            from sklearn.metrics import (accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         confusion_matrix,
                                         classification_report,
                                         roc_curve, auc)

            cm = confusion_matrix(y_true, y_pred)

            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(n_classes):
                y_bin = (y_true == i).astype(int)
                fpr[i], tpr[i], _ = roc_curve(y_bin, y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            y_bin_micro = np.eye(n_classes)[y_true]
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_bin_micro.ravel(), y_scores.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            roc_data = {
                'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
                'n_classes': n_classes, 'class_names': class_names,
            }

            accuracy = accuracy_score(y_true, y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision_per = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
            report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

            per_class_metrics = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_per': list(precision_per),
                'recall_per': list(recall_per),
                'f1_per': list(f1_per),
                'class_names': class_names,
            }

            self.log_signal.emit('\n========== 多细胞检测 + 分类验证 ==========')
            self.log_signal.emit(f'验证图像数: {len(image_files)}')
            self.log_signal.emit(f'IoU 匹配阈值: {self.iou_thresh}')
            self.log_signal.emit(f'匹配细胞对数: {len(y_true)}')
            self.log_signal.emit(f'真实标注总数: {sum(total_gt.values())}')
            self.log_signal.emit(f'检测预测总数: {sum(total_pred.values())}')
            self.log_signal.emit('')

            for cls_name in class_names:
                cn_name = CLASS_NAMES_CN[CLASS_NAMES.index(cls_name)]
                self.log_signal.emit(
                    f'  {cn_name} ({cls_name}): '
                    f'真实={total_gt[cls_name]}, 检测={total_pred[cls_name]}, '
                    f'TP={total_tp.get(cls_name, 0)}'
                )

            self.log_signal.emit(f'\n[整体] accuracy={accuracy:.4f}  '
                                 f'precision={precision_macro:.4f}  '
                                 f'recall={recall_macro:.4f}  '
                                 f'f1={f1_macro:.4f}')
            self.log_signal.emit(f'\n{report}')

            self.log_signal.emit('\n========== 逐图像详情 ==========')
            for r in image_results:
                gt_str = ' '.join([f'{k}={v}' for k, v in r['gt'].items()])
                pred_str = ' '.join([f'{k}={v}' for k, v in r['pred'].items()])
                fp_str = ' '.join([f'{k}={v}' for k, v in r['fp'].items()])
                fn_str = ' '.join([f'{k}={v}' for k, v in r['fn'].items()])
                status = '[有标注]' if r['has_gt'] else '[无标注]'
                self.log_signal.emit(f'  {r["name"]} {status}  匹配: {r["matched"]}')
                self.log_signal.emit(f'    真实: {gt_str}')
                self.log_signal.emit(f'    预测: {pred_str}')
                self.log_signal.emit(f'    FP: {fp_str}   FN: {fn_str}')

            self.log_signal.emit('\n多细胞验证完成！')
            self.progress_signal.emit(100)
            self.finished_signal.emit(True, '', cm, roc_data, class_names, per_class_metrics)

        except Exception as e:
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, f'多细胞验证失败: {str(e)}', None, None, None, None)


def _load_image_pixmap(file_path):
    stream = open(file_path, 'rb')
    buf = np.frombuffer(stream.read(), dtype=np.uint8)
    stream.close()
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    _, png_data = cv2.imencode('.png', img)
    pixmap = QPixmap()
    pixmap.loadFromData(png_data.tobytes())
    return pixmap if not pixmap.isNull() else None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image_path = None
        self.detect_image_path = None
        self.transform_test = None

        self.init_ui()
        self.init_model()

    def init_ui(self):
        self.setWindowTitle('血液细胞识别系统 - Blood Cell Recognition')
        self.setGeometry(100, 100, 1280, 800)

        self.setStyleSheet('''
            QMainWindow { background-color: #f0f2f5; }
            QGroupBox {
                font-weight: bold; border: 1px solid #d0d0d0;
                border-radius: 6px; margin-top: 12px; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
            }
            QPushButton {
                background-color: #1890ff; color: white; border: none;
                border-radius: 4px; padding: 6px 16px; min-height: 28px;
            }
            QPushButton:hover { background-color: #40a9ff; }
            QPushButton:pressed { background-color: #096dd9; }
            QPushButton:disabled { background-color: #d9d9d9; color: #999; }
            QTabWidget::pane { border: 1px solid #d0d0d0; background: white; }
            QTabBar::tab {
                background: #e8e8e8; padding: 8px 20px; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: white; font-weight: bold; }
            QTextEdit { border: 1px solid #d0d0d0; border-radius: 4px; }
        ''')

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel('血液细胞智能识别系统')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 20px; font-weight: bold; color: #1890ff; padding: 8px;')
        layout.addWidget(title)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self.create_predict_tab(), '📷 细胞识别')
        self.tabs.addTab(self.create_detect_tab(), '🔬 多细胞检测')
        self.tabs.addTab(self.create_train_tab(), '🏋️ 模型训练')
        self.tabs.addTab(self.create_validate_tab(), '📊 模型验证')
        self.tabs.addTab(self.create_convert_tab(), '🔄 模型转换')

    def init_model(self):
        ok, torch, transforms, ResNet = _check_torch()
        if not ok:
            self.status_label.setText('PyTorch 未加载，模型功能不可用')
            return
        try:
            if os.path.exists(opt.test_model_path):
                self.model = ResNet(num_classes=3, pretrained=False)
                ckpt = torch.load(opt.test_model_path, map_location='cpu')
                self.model.load_state_dict(ckpt, strict=False)
                self.model.eval()

                self.transform_test = transforms.Compose([
                    transforms.Resize((opt.loadsize, opt.loadsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.6786, 0.6413, 0.6605], [0.2599, 0.2595, 0.2569])
                ])
                self.status_label.setText('模型已加载: best.pth')
            else:
                self.status_label.setText('未找到模型文件，请先训练')
        except Exception as e:
            self.status_label.setText(f'模型加载失败: {str(e)}')

    def create_predict_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_bar = QHBoxLayout()

        self.status_label = QLabel('就绪')
        self.status_label.setStyleSheet('color: #666;')
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()

        self.load_model_btn = QPushButton('加载模型')
        self.load_model_btn.clicked.connect(self.load_model)
        top_bar.addWidget(self.load_model_btn)

        self.select_img_btn = QPushButton('选择图像')
        self.select_img_btn.clicked.connect(self.select_image)
        top_bar.addWidget(self.select_img_btn)

        self.predict_btn = QPushButton('开始识别')
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        top_bar.addWidget(self.predict_btn)

        layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel('原始图像'))
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setStyleSheet('border: 1px solid #d0d0d0; background: #fafafa;')
        self.original_label.setText('请选择图像')
        left_layout.addWidget(self.original_label)

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        result_group = QGroupBox('识别结果')
        result_layout = QGridLayout(result_group)

        self.class_labels = []
        self.progress_bars = []
        self.conf_labels = []

        for i, (en, cn) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_CN)):
            lbl = QLabel(f'{cn} ({en})')
            lbl.setStyleSheet('font-weight: bold;')
            result_layout.addWidget(lbl, i, 0)

            pb = QProgressBar()
            pb.setMaximum(100)
            pb.setTextVisible(False)
            pb.setStyleSheet('QProgressBar { border: 1px solid #d0d0d0; border-radius: 3px; }')
            result_layout.addWidget(pb, i, 1)

            cl = QLabel('0%')
            result_layout.addWidget(cl, i, 2)

            self.class_labels.append(lbl)
            self.progress_bars.append(pb)
            self.conf_labels.append(cl)

        self.result_image_label = QLabel()
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumSize(200, 200)
        self.result_image_label.setStyleSheet('border: 1px solid #d0d0d0; background: #fafafa;')
        self.result_image_label.setText('识别结果预览')

        right_layout.addWidget(result_group)
        right_layout.addWidget(self.result_image_label)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        layout.addWidget(splitter)

        return tab

    def create_detect_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_bar = QHBoxLayout()

        self.detect_status_label = QLabel('请先训练或加载模型，然后选择图像')
        self.detect_status_label.setStyleSheet('color: #666;')
        top_bar.addWidget(self.detect_status_label)
        top_bar.addStretch()

        self.detect_load_model_btn = QPushButton('加载模型')
        self.detect_load_model_btn.clicked.connect(self.load_model)
        top_bar.addWidget(self.detect_load_model_btn)

        self.detect_select_btn = QPushButton('选择多细胞图像')
        self.detect_select_btn.clicked.connect(self.select_detect_image)
        top_bar.addWidget(self.detect_select_btn)

        self.detect_btn = QPushButton('🔍 开始检测')
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        top_bar.addWidget(self.detect_btn)

        layout.addLayout(top_bar)

        config_bar = QHBoxLayout()
        config_bar.addWidget(QLabel('最小面积:'))
        self.detect_min_area = QSpinBox()
        self.detect_min_area.setRange(10, 10000)
        self.detect_min_area.setValue(200)
        self.detect_min_area.setToolTip('小于此面积的轮廓将被忽略')
        config_bar.addWidget(self.detect_min_area)

        config_bar.addWidget(QLabel('最大面积:'))
        self.detect_max_area = QSpinBox()
        self.detect_max_area.setRange(100, 200000)
        self.detect_max_area.setValue(50000)
        self.detect_max_area.setToolTip('大于此面积的轮廓将被忽略')
        config_bar.addWidget(self.detect_max_area)
        config_bar.addStretch()
        layout.addLayout(config_bar)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel('原始图像'))
        self.detect_original_label = QLabel()
        self.detect_original_label.setAlignment(Qt.AlignCenter)
        self.detect_original_label.setMinimumSize(400, 350)
        self.detect_original_label.setStyleSheet(
            'border: 1px solid #d0d0d0; background: #fafafa;'
        )
        self.detect_original_label.setText('请选择一张含多细胞的血液涂片图像')
        left_layout.addWidget(self.detect_original_label)

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel('检测结果'))
        self.detect_result_label = QLabel()
        self.detect_result_label.setAlignment(Qt.AlignCenter)
        self.detect_result_label.setMinimumSize(400, 350)
        self.detect_result_label.setStyleSheet(
            'border: 1px solid #d0d0d0; background: #fafafa;'
        )
        self.detect_result_label.setText('检测结果将在此显示')
        right_layout.addWidget(self.detect_result_label)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        layout.addWidget(splitter)

        summary_group = QGroupBox('计数统计')
        summary_layout = QHBoxLayout(summary_group)

        self.detect_count_labels = {}
        colors_hex = ['#0000FF', '#00FF00', '#FF0000']
        for (en, cn), color in zip(
            zip(CLASS_NAMES, CLASS_NAMES_CN), colors_hex
        ):
            lbl = QLabel(f'{cn} ({en}): 0')
            lbl.setStyleSheet(
                f'font-size: 14px; font-weight: bold; color: {color}; '
                'padding: 8px 16px;'
            )
            self.detect_count_labels[en] = lbl
            summary_layout.addWidget(lbl)

        layout.addWidget(summary_group)

        self.detect_log = QTextEdit()
        self.detect_log.setReadOnly(True)
        self.detect_log.setMaximumHeight(100)
        self.detect_log.setStyleSheet(
            'background: #1e1e1e; color: #d4d4d4; font-family: Consolas;'
        )
        layout.addWidget(self.detect_log)

        return tab

    def select_detect_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择多细胞血液涂片图像', './datasets/',
            'Images (*.jpg *.jpeg *.png *.bmp *.tiff)',
            options=QFileDialog.DontUseNativeDialog
        )
        if not file_path:
            return
        pixmap = _load_image_pixmap(file_path)
        if pixmap is None:
            QMessageBox.warning(self, '错误', '无法读取图像文件')
            return
        self.detect_image_path = file_path
        label_size = self.detect_original_label.size()
        if label_size.width() > 10 and label_size.height() > 10:
            scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
        else:
            scaled = pixmap
        self.detect_original_label.setPixmap(scaled)
        self.detect_status_label.setText(
            f'已选择: {os.path.basename(file_path)}'
        )
        if self.model is not None:
            self.detect_btn.setEnabled(True)

    def run_detection(self):
        if self.model is None:
            QMessageBox.warning(self, '提示', '请先训练模型或加载已有模型')
            return
        if not self.detect_image_path:
            QMessageBox.warning(self, '提示', '请先选择一张图像')
            return

        self.detect_btn.setEnabled(False)
        self.detect_select_btn.setEnabled(False)
        self.detect_status_label.setText('正在检测细胞...')
        self.detect_log.clear()

        self.detect_thread = DetectThread(
            self.detect_image_path, self.model, self.transform_test,
            opt.loadsize,
            self.detect_min_area.value(), self.detect_max_area.value()
        )
        self.detect_thread.log_signal.connect(self.detect_log.append)
        self.detect_thread.finished_signal.connect(self.on_detection_finished)
        self.detect_thread.start()

    def on_detection_finished(self, success, msg, result_img, counts):
        self.detect_btn.setEnabled(True)
        self.detect_select_btn.setEnabled(True)

        if not success:
            self.detect_status_label.setText('检测失败')
            self.detect_log.append(f'\n❌ {msg}')
            QMessageBox.critical(self, '检测失败', msg)
            return

        if result_img is not None:
            _, png_data = cv2.imencode('.png', result_img)
            pixmap = QPixmap()
            pixmap.loadFromData(png_data.tobytes())
            scaled = pixmap.scaled(
                self.detect_result_label.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.detect_result_label.setPixmap(scaled)

        if counts is not None:
            total = sum(counts.values())
            for en in CLASS_NAMES:
                cn = CLASS_NAMES_CN[CLASS_NAMES.index(en)]
                self.detect_count_labels[en].setText(
                    f'{cn} ({en}): {counts[en]}'
                )
            self.detect_status_label.setText(f'检测完成: 共 {total} 个细胞')

    def create_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        config_group = QGroupBox('训练参数配置')
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel('设备:'), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cpu', 'cuda', 'mps'])
        self.device_combo.setCurrentText('cpu')
        config_layout.addWidget(self.device_combo, 0, 1)

        config_layout.addWidget(QLabel('训练轮数:'), 0, 2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(3)
        config_layout.addWidget(self.epochs_spin, 0, 3)

        config_layout.addWidget(QLabel('批次大小:'), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        config_layout.addWidget(self.batch_spin, 1, 1)

        config_layout.addWidget(QLabel('学习率:'), 1, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(0.01)
        config_layout.addWidget(self.lr_spin, 1, 3)

        config_layout.addWidget(QLabel('图像尺寸:'), 2, 0)
        self.loadsize_spin = QSpinBox()
        self.loadsize_spin.setRange(32, 1024)
        self.loadsize_spin.setValue(224)
        config_layout.addWidget(self.loadsize_spin, 2, 1)

        layout.addWidget(config_group)

        path_group = QGroupBox('路径配置')
        path_layout = QGridLayout(path_group)

        path_layout.addWidget(QLabel('训练集:'), 0, 0)
        self.train_path_edit = QLineEdit(opt.dataset_train)
        path_layout.addWidget(self.train_path_edit, 0, 1)

        path_layout.addWidget(QLabel('验证集:'), 1, 0)
        self.val_path_edit = QLineEdit(opt.dataset_val)
        path_layout.addWidget(self.val_path_edit, 1, 1)

        path_layout.addWidget(QLabel('测试集:'), 2, 0)
        self.test_path_edit = QLineEdit(opt.dataset_test)
        path_layout.addWidget(self.test_path_edit, 2, 1)

        path_layout.addWidget(QLabel('模型保存:'), 3, 0)
        self.ckpt_path_edit = QLineEdit(opt.checkpoints)
        path_layout.addWidget(self.ckpt_path_edit, 3, 1)

        layout.addWidget(path_group)

        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton('🚀 开始训练')
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet('font-size: 14px; padding: 10px 30px;')
        btn_layout.addStretch()
        btn_layout.addWidget(self.train_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        layout.addWidget(self.train_progress)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet('background: #1e1e1e; color: #d4d4d4; font-family: Consolas;')
        layout.addWidget(self.train_log)

        return tab

    def create_validate_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        config_group = QGroupBox('验证配置')
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel('模型路径:'), 0, 0)
        self.val_model_edit = QLineEdit(opt.test_model_path)
        config_layout.addWidget(self.val_model_edit, 0, 1)
        browse_val_model_btn = QPushButton('浏览')
        browse_val_model_btn.clicked.connect(self.browse_val_model)
        config_layout.addWidget(browse_val_model_btn, 0, 2)

        config_layout.addWidget(QLabel('验证集路径:'), 1, 0)
        self.val_data_edit = QLineEdit(opt.dataset_val)
        config_layout.addWidget(self.val_data_edit, 1, 1)
        browse_val_data_btn = QPushButton('浏览')
        browse_val_data_btn.clicked.connect(self.browse_val_data)
        config_layout.addWidget(browse_val_data_btn, 1, 2)

        config_layout.addWidget(QLabel('输入尺寸:'), 2, 0)
        self.val_loadsize_spin = QSpinBox()
        self.val_loadsize_spin.setRange(32, 1024)
        self.val_loadsize_spin.setValue(224)
        config_layout.addWidget(self.val_loadsize_spin, 2, 1)

        layout.addWidget(config_group)

        btn_layout = QHBoxLayout()
        self.validate_btn = QPushButton('🔍 分类验证')
        self.validate_btn.clicked.connect(self.start_validation)
        self.validate_btn.setStyleSheet('font-size: 14px; padding: 10px 20px;')
        btn_layout.addStretch()
        btn_layout.addWidget(self.validate_btn)

        self.multi_cell_val_btn = QPushButton('🔬 多细胞验证')
        self.multi_cell_val_btn.clicked.connect(self.start_multi_cell_validation)
        self.multi_cell_val_btn.setStyleSheet(
            'font-size: 14px; padding: 10px 20px; '
            'background-color: #52c41a;'
        )
        self.multi_cell_val_btn.setToolTip('对 /datasets/val/ 中的多细胞图像进行检测验证')
        btn_layout.addWidget(self.multi_cell_val_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        multi_cell_config = QHBoxLayout()
        multi_cell_config.addWidget(QLabel('多细胞Val目录:'))
        self.multi_val_dir_edit = QLineEdit('./datasets/val')
        multi_cell_config.addWidget(self.multi_val_dir_edit)
        browse_multi_val_btn = QPushButton('浏览')
        browse_multi_val_btn.clicked.connect(self.browse_multi_val_data)
        multi_cell_config.addWidget(browse_multi_val_btn)

        multi_cell_config.addWidget(QLabel('最小面积:'))
        self.multi_val_min_area = QSpinBox()
        self.multi_val_min_area.setRange(10, 10000)
        self.multi_val_min_area.setValue(200)
        multi_cell_config.addWidget(self.multi_val_min_area)

        multi_cell_config.addWidget(QLabel('最大面积:'))
        self.multi_val_max_area = QSpinBox()
        self.multi_val_max_area.setRange(100, 200000)
        self.multi_val_max_area.setValue(50000)
        multi_cell_config.addWidget(self.multi_val_max_area)
        layout.addLayout(multi_cell_config)

        self.val_progress = QProgressBar()
        self.val_progress.setVisible(False)
        layout.addWidget(self.val_progress)

        splitter = QSplitter(Qt.Vertical)

        fig_panel = QTabWidget()
        fig_panel.setMinimumHeight(450)

        cm_widget = QWidget()
        cm_layout = QVBoxLayout(cm_widget)
        self.cm_canvas_layout = QVBoxLayout()
        cm_layout.addLayout(self.cm_canvas_layout)
        cm_layout.addStretch()
        fig_panel.addTab(cm_widget, '混淆矩阵')

        roc_widget = QWidget()
        roc_layout = QVBoxLayout(roc_widget)
        self.roc_canvas_layout = QVBoxLayout()
        roc_layout.addLayout(self.roc_canvas_layout)
        roc_layout.addStretch()
        fig_panel.addTab(roc_widget, 'ROC曲线')

        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        self.metrics_canvas_layout = QVBoxLayout()
        metrics_layout.addLayout(self.metrics_canvas_layout)
        metrics_layout.addStretch()
        fig_panel.addTab(metrics_widget, '其他评估指标')

        splitter.addWidget(fig_panel)

        self.val_log = QTextEdit()
        self.val_log.setReadOnly(True)
        self.val_log.setStyleSheet(
            'background: #1e1e1e; color: #d4d4d4; font-family: Consolas;'
        )
        self.val_log.setMinimumHeight(150)
        splitter.addWidget(self.val_log)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

        return tab

    def browse_val_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型文件', './checkpoints/', 'Model Files (*.pth *.pt)'
        )
        if path:
            self.val_model_edit.setText(path)

    def browse_val_data(self):
        path = QFileDialog.getExistingDirectory(
            self, '选择验证集目录', './datasets/'
        )
        if path:
            self.val_data_edit.setText(path)

    def browse_multi_val_data(self):
        path = QFileDialog.getExistingDirectory(
            self, '选择多细胞验证目录', './datasets/'
        )
        if path:
            self.multi_val_dir_edit.setText(path)

    def start_validation(self):
        model_path = self.val_model_edit.text()
        val_data_path = self.val_data_edit.text()
        loadsize = self.val_loadsize_spin.value()

        if not os.path.exists(model_path):
            QMessageBox.warning(self, '错误', '模型文件不存在')
            return
        if not os.path.isdir(val_data_path):
            QMessageBox.warning(self, '错误', '验证集目录不存在')
            return

        self.validate_btn.setEnabled(False)
        self.multi_cell_val_btn.setEnabled(False)
        self.val_progress.setVisible(True)
        self.val_progress.setValue(0)
        self.val_log.clear()

        self.val_thread = ValidateThread(model_path, val_data_path, loadsize)
        self.val_thread.log_signal.connect(self.val_log.append)
        self.val_thread.progress_signal.connect(self.val_progress.setValue)
        self.val_thread.finished_signal.connect(self.on_validation_finished)
        self.val_thread.start()

    def start_multi_cell_validation(self):
        if self.model is None:
            QMessageBox.warning(self, '提示', '请先加载模型')
            return

        val_dir = self.multi_val_dir_edit.text()
        if not os.path.isdir(val_dir):
            QMessageBox.warning(self, '错误', '多细胞验证目录不存在')
            return

        self.validate_btn.setEnabled(False)
        self.multi_cell_val_btn.setEnabled(False)
        self.val_progress.setVisible(True)
        self.val_progress.setValue(0)
        self.val_log.clear()
        self.val_log.append('开始多细胞图像验证...')

        self.multi_val_thread = MultiCellValThread(
            val_dir, self.model, self.transform_test, self.val_loadsize_spin.value(),
            self.multi_val_min_area.value(), self.multi_val_max_area.value()
        )
        self.multi_val_thread.log_signal.connect(self.val_log.append)
        self.multi_val_thread.progress_signal.connect(self.val_progress.setValue)
        self.multi_val_thread.finished_signal.connect(self.on_multi_val_finished)
        self.multi_val_thread.start()

    def on_multi_val_finished(self, success, msg, conf_matrix=None, roc_data=None,
                                class_names=None, per_class_metrics=None):
        self.validate_btn.setEnabled(True)
        self.multi_cell_val_btn.setEnabled(True)
        if not success:
            self.val_log.append(f'\n❌ {msg}')
            QMessageBox.critical(self, '多细胞验证失败', msg)
            return

        self.val_log.append('\n✅ 多细胞验证完成！')

        self._clear_layout(self.cm_canvas_layout)
        if conf_matrix is not None and class_names is not None:
            cm_fig = Figure(figsize=(5, 4), dpi=100)
            cm_ax = cm_fig.add_subplot(111)
            im = cm_ax.imshow(conf_matrix, cmap='Blues')
            cm_ax.set_title('多细胞检测 - Confusion Matrix / 混淆矩阵')
            cm_ax.set_xticks(range(len(class_names)))
            cm_ax.set_yticks(range(len(class_names)))
            cm_ax.set_xticklabels(class_names)
            cm_ax.set_yticklabels(class_names)
            cm_ax.set_xlabel('Predicted Label / 预测标签')
            cm_ax.set_ylabel('True Label / 真实标签')
            cm_fig.colorbar(im, ax=cm_ax)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    cm_ax.text(j, i, str(conf_matrix[i, j]),
                               ha='center', va='center',
                               color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
            cm_canvas = FigureCanvas(cm_fig)
            self.cm_canvas_layout.addWidget(cm_canvas)

        self._clear_layout(self.roc_canvas_layout)
        if roc_data is not None:
            roc_fig = Figure(figsize=(5, 4), dpi=100)
            roc_ax = roc_fig.add_subplot(111)
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            roc_auc = roc_data['roc_auc']
            n_cls = roc_data['n_classes']
            cn = roc_data['class_names']
            roc_ax.plot(fpr["micro"], tpr["micro"],
                        label=f'micro (AUC={roc_auc["micro"]:.2f})',
                        color='deeppink', linestyle=':', linewidth=3)
            roc_ax.plot(fpr["macro"], tpr["macro"],
                        label=f'macro (AUC={roc_auc["macro"]:.2f})',
                        color='navy', linestyle=':', linewidth=3)
            colors = ['blue', 'green', 'red']
            for i in range(n_cls):
                roc_ax.plot(fpr[i], tpr[i], color=colors[i], linewidth=2,
                            label=f'{cn[i]} (AUC={roc_auc[i]:.2f})')
            roc_ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            roc_ax.set_xlim([0.0, 1.0])
            roc_ax.set_ylim([0.0, 1.05])
            roc_ax.set_xlabel('False Positive Rate / 假阳性率')
            roc_ax.set_ylabel('True Positive Rate / 真阳性率')
            roc_ax.set_title('多细胞检测 - ROC Curve / ROC 曲线')
            roc_ax.legend(loc='lower right', fontsize=8)
            roc_canvas = FigureCanvas(roc_fig)
            self.roc_canvas_layout.addWidget(roc_canvas)

        self._clear_layout(self.metrics_canvas_layout)
        if per_class_metrics is not None:
            metrics_fig = Figure(figsize=(9, 5), dpi=100)
            ax1 = metrics_fig.add_subplot(131)
            ax2 = metrics_fig.add_subplot(132)
            ax3 = metrics_fig.add_subplot(133)
            cn_names = per_class_metrics['class_names']
            x_pos = range(len(cn_names))
            bar_colors = ['#5470c6', '#91cc75', '#ee6666']
            bar_width = 0.5

            precision_vals = per_class_metrics['precision_per']
            bars1 = ax1.bar(x_pos, precision_vals, bar_width, color=bar_colors)
            ax1.set_title('Precision / 精确率')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(cn_names, fontsize=8)
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel('Precision / 精确率')
            for bar, val in zip(bars1, precision_vals):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            recall_vals = per_class_metrics['recall_per']
            bars2 = ax2.bar(x_pos, recall_vals, bar_width, color=bar_colors)
            ax2.set_title('Recall / 召回率')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(cn_names, fontsize=8)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel('Recall / 召回率')
            for bar, val in zip(bars2, recall_vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            f1_vals = per_class_metrics['f1_per']
            bars3 = ax3.bar(x_pos, f1_vals, bar_width, color=bar_colors)
            ax3.set_title('F1-Score / F1分数')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(cn_names, fontsize=8)
            ax3.set_ylim(0, 1.05)
            ax3.set_ylabel('F1-Score / F1分数')
            for bar, val in zip(bars3, f1_vals):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            metrics_fig.suptitle(
                f'多细胞检测 - Accuracy/准确率={per_class_metrics["accuracy"]:.4f}  '
                f'Macro-P/宏精确率={per_class_metrics["precision_macro"]:.4f}  '
                f'Macro-R/宏召回率={per_class_metrics["recall_macro"]:.4f}  '
                f'Macro-F1/宏F1={per_class_metrics["f1_macro"]:.4f}',
                fontsize=10, fontweight='bold'
            )
            metrics_fig.tight_layout(rect=[0, 0, 1, 0.92])
            metrics_canvas = FigureCanvas(metrics_fig)
            self.metrics_canvas_layout.addWidget(metrics_canvas)

    def on_validation_finished(self, success, msg, conf_matrix, roc_data, class_names, per_class_metrics=None):
        self.validate_btn.setEnabled(True)
        self.multi_cell_val_btn.setEnabled(True)
        if not success:
            self.val_log.append(f'\n❌ {msg}')
            QMessageBox.critical(self, '验证失败', msg)
            return

        self.val_log.append(f'\n✅ {msg}')

        self._clear_layout(self.cm_canvas_layout)
        if conf_matrix is not None:
            cm_fig = Figure(figsize=(5, 4), dpi=100)
            cm_ax = cm_fig.add_subplot(111)
            im = cm_ax.imshow(conf_matrix, cmap='Blues')
            cm_ax.set_title('Confusion Matrix / 混淆矩阵')
            cm_ax.set_xticks(range(len(class_names)))
            cm_ax.set_yticks(range(len(class_names)))
            cm_ax.set_xticklabels(class_names)
            cm_ax.set_yticklabels(class_names)
            cm_ax.set_xlabel('Predicted Label / 预测标签')
            cm_ax.set_ylabel('True Label / 真实标签')
            cm_fig.colorbar(im, ax=cm_ax)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    cm_ax.text(j, i, str(conf_matrix[i, j]),
                               ha='center', va='center',
                               color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
            cm_canvas = FigureCanvas(cm_fig)
            self.cm_canvas_layout.addWidget(cm_canvas)

        self._clear_layout(self.roc_canvas_layout)
        if roc_data is not None:
            roc_fig = Figure(figsize=(5, 4), dpi=100)
            roc_ax = roc_fig.add_subplot(111)

            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            roc_auc = roc_data['roc_auc']
            n_classes = roc_data['n_classes']
            cn = roc_data['class_names']

            roc_ax.plot(fpr["micro"], tpr["micro"],
                        label=f'micro (AUC={roc_auc["micro"]:.2f})',
                        color='deeppink', linestyle=':', linewidth=3)
            roc_ax.plot(fpr["macro"], tpr["macro"],
                        label=f'macro (AUC={roc_auc["macro"]:.2f})',
                        color='navy', linestyle=':', linewidth=3)

            colors = ['blue', 'green', 'red']
            for i in range(n_classes):
                roc_ax.plot(fpr[i], tpr[i], color=colors[i], linewidth=2,
                            label=f'{cn[i]} (AUC={roc_auc[i]:.2f})')

            roc_ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            roc_ax.set_xlim([0.0, 1.0])
            roc_ax.set_ylim([0.0, 1.05])
            roc_ax.set_xlabel('False Positive Rate / 假阳性率')
            roc_ax.set_ylabel('True Positive Rate / 真阳性率')
            roc_ax.set_title('ROC Curve - Multi-class / ROC 曲线')
            roc_ax.legend(loc='lower right', fontsize=8)
            roc_canvas = FigureCanvas(roc_fig)
            self.roc_canvas_layout.addWidget(roc_canvas)

        self._clear_layout(self.metrics_canvas_layout)
        if per_class_metrics is not None:
            metrics_fig = Figure(figsize=(9, 5), dpi=100)

            ax1 = metrics_fig.add_subplot(131)
            ax2 = metrics_fig.add_subplot(132)
            ax3 = metrics_fig.add_subplot(133)

            cn_names = per_class_metrics['class_names']
            x_pos = range(len(cn_names))
            bar_colors = ['#5470c6', '#91cc75', '#ee6666']
            bar_width = 0.5

            precision_vals = per_class_metrics['precision_per']
            bars1 = ax1.bar(x_pos, precision_vals, bar_width, color=bar_colors)
            ax1.set_title('Precision / 精确率')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(cn_names, fontsize=8)
            ax1.set_ylim(0, 1.05)
            ax1.set_ylabel('Precision / 精确率')
            for bar, val in zip(bars1, precision_vals):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            recall_vals = per_class_metrics['recall_per']
            bars2 = ax2.bar(x_pos, recall_vals, bar_width, color=bar_colors)
            ax2.set_title('Recall / 召回率')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(cn_names, fontsize=8)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel('Recall / 召回率')
            for bar, val in zip(bars2, recall_vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            f1_vals = per_class_metrics['f1_per']
            bars3 = ax3.bar(x_pos, f1_vals, bar_width, color=bar_colors)
            ax3.set_title('F1-Score / F1分数')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(cn_names, fontsize=8)
            ax3.set_ylim(0, 1.05)
            ax3.set_ylabel('F1-Score / F1分数')
            for bar, val in zip(bars3, f1_vals):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            metrics_fig.suptitle(
                f'Overall: Accuracy/准确率={per_class_metrics["accuracy"]:.4f}  '
                f'Macro-P/宏精确率={per_class_metrics["precision_macro"]:.4f}  '
                f'Macro-R/宏召回率={per_class_metrics["recall_macro"]:.4f}  '
                f'Macro-F1/宏F1={per_class_metrics["f1_macro"]:.4f}',
                fontsize=10, fontweight='bold'
            )
            metrics_fig.tight_layout(rect=[0, 0, 1, 0.92])
            metrics_canvas = FigureCanvas(metrics_fig)
            self.metrics_canvas_layout.addWidget(metrics_canvas)

    @staticmethod
    def _clear_layout(layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def create_convert_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        pth_group = QGroupBox('模型转换: PTH → ONNX')
        pth_layout = QGridLayout(pth_group)

        pth_layout.addWidget(QLabel('PTH 模型路径:'), 0, 0)
        self.pth_path_edit = QLineEdit(opt.test_model_path)
        pth_layout.addWidget(self.pth_path_edit, 0, 1)
        browse_pth_btn = QPushButton('浏览')
        browse_pth_btn.clicked.connect(self.browse_pth)
        pth_layout.addWidget(browse_pth_btn, 0, 2)

        pth_layout.addWidget(QLabel('ONNX 输出路径:'), 1, 0)
        self.onnx_path_edit = QLineEdit(opt.onnx_path)
        pth_layout.addWidget(self.onnx_path_edit, 1, 1)
        browse_onnx_btn = QPushButton('浏览')
        browse_onnx_btn.clicked.connect(self.browse_onnx)
        pth_layout.addWidget(browse_onnx_btn, 1, 2)

        pth_layout.addWidget(QLabel('输入尺寸:'), 2, 0)
        self.convert_loadsize = QSpinBox()
        self.convert_loadsize.setRange(32, 1024)
        self.convert_loadsize.setValue(224)
        pth_layout.addWidget(self.convert_loadsize, 2, 1)

        layout.addWidget(pth_group)

        convert_btn_layout = QHBoxLayout()
        self.convert_btn = QPushButton('🔄 开始转换')
        self.convert_btn.clicked.connect(self.convert_model)
        self.convert_btn.setStyleSheet('font-size: 14px; padding: 10px 30px;')
        convert_btn_layout.addStretch()
        convert_btn_layout.addWidget(self.convert_btn)
        convert_btn_layout.addStretch()
        layout.addLayout(convert_btn_layout)

        self.convert_log = QTextEdit()
        self.convert_log.setReadOnly(True)
        self.convert_log.setStyleSheet('background: #1e1e1e; color: #d4d4d4; font-family: Consolas;')
        layout.addWidget(self.convert_log)

        return tab

    def load_model(self):
        ok, torch, transforms, ResNet = _check_torch()
        if not ok:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择模型文件', './checkpoints/', 'Model Files (*.pth *.pt)'
        )
        if file_path:
            try:
                self.model = ResNet(num_classes=3, pretrained=False)
                ckpt = torch.load(file_path, map_location='cpu')
                self.model.load_state_dict(ckpt, strict=False)
                self.model.eval()

                self.transform_test = transforms.Compose([
                    transforms.Resize((opt.loadsize, opt.loadsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.6786, 0.6413, 0.6605], [0.2599, 0.2595, 0.2569])
                ])

                model_name = os.path.basename(file_path)
                self.status_label.setText(f'模型已加载: {model_name}')
                self.detect_status_label.setText(f'模型已加载: {model_name}')
                if self.current_image_path:
                    self.predict_btn.setEnabled(True)
                if self.detect_image_path:
                    self.detect_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.warning(self, '加载失败', str(e))

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择血液细胞图像', './datasets/',
            'Images (*.jpg *.jpeg *.png *.bmp *.tiff)'
        )
        if file_path:
            pixmap = _load_image_pixmap(file_path)
            if pixmap is None:
                QMessageBox.warning(self, '错误', '无法读取图像文件')
                return
            self.current_image_path = file_path
            scaled = pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_label.setPixmap(scaled)
            self.status_label.setText(f'已选择: {os.path.basename(file_path)}')

            if self.model is not None:
                self.predict_btn.setEnabled(True)

    def predict(self):
        if self.model is None or self.current_image_path is None:
            return

        self.predict_btn.setEnabled(False)
        self.select_img_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.status_label.setText('正在识别...')

        self.predict_thread = PredictThread(
            self.model, self.current_image_path, self.transform_test, opt.loadsize
        )
        self.predict_thread.result_signal.connect(self.on_predict_result)
        self.predict_thread.start()

    def on_predict_result(self, pred_class, confidence, original):
        self.predict_btn.setEnabled(True)
        self.select_img_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)

        if pred_class == -1:
            self.status_label.setText('识别失败')
            return

        colors_bgr = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255)
        ]
        colors_rgb = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0)
        ]

        for i in range(3):
            self.progress_bars[i].setValue(0)
            self.conf_labels[i].setText('0%')
            self.progress_bars[i].setStyleSheet(
                'QProgressBar { border: 1px solid #d0d0d0; border-radius: 3px; }'
                'QProgressBar::chunk { background-color: #d9d9d9; }'
            )

        for i in range(3):
            self.progress_bars[i].setValue(0)
            self.conf_labels[i].setText(f'{confidence * 100:.1f}%' if i == pred_class else '0%')

        bar_style = (
            f'QProgressBar {{ border: 1px solid #d0d0d0; border-radius: 3px; }}'
            f'QProgressBar::chunk {{ background-color: rgb{colors_rgb[pred_class]}; }}'
        )
        self.progress_bars[pred_class].setValue(int(confidence * 100))
        self.progress_bars[pred_class].setStyleSheet(bar_style)

        self.class_labels[pred_class].setStyleSheet(
            f'font-weight: bold; color: rgb{colors_rgb[pred_class]}; font-size: 13px;'
        )

        self.status_label.setText(
            f'识别结果: {CLASS_NAMES_CN[pred_class]} ({CLASS_NAMES[pred_class]}) 置信度: {confidence:.2%}'
        )

        if original is not None:
            h, w = original.shape[:2]
            cv2.rectangle(original, (10, 10), (w - 10, h - 10), colors_bgr[pred_class], 3)
            text = f'{CLASS_NAMES_CN[pred_class]}: {confidence:.2%}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original, text, (20, 40), font, 1, colors_bgr[pred_class], 2)

            _, png_data = cv2.imencode('.png', original)
            pixmap = QPixmap()
            pixmap.loadFromData(png_data.tobytes())
            scaled = pixmap.scaled(self.result_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.result_image_label.setPixmap(scaled)

    def start_training(self):
        args_dict = {
            'device': self.device_combo.currentText(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'lr': self.lr_spin.value(),
            'loadsize': self.loadsize_spin.value(),
            'dataset_train': self.train_path_edit.text(),
            'dataset_val': self.val_path_edit.text(),
            'dataset_test': self.test_path_edit.text(),
            'checkpoints': self.ckpt_path_edit.text(),
            'log_dir': './log_dir',
            'logging_txt': './log_dir/logging.txt',
        }

        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)
        self.train_log.clear()

        self.train_thread = TrainThread(args_dict)
        self.train_thread.log_signal.connect(self.on_train_log)
        self.train_thread.progress_signal.connect(self.train_progress.setValue)
        self.train_thread.finished_signal.connect(self.on_train_finished)
        self.train_thread.start()

    def on_train_log(self, msg):
        self.train_log.append(msg)

    def on_train_finished(self, success, msg):
        self.train_btn.setEnabled(True)
        if success:
            self.train_log.append(f'\n✅ {msg}')
            QMessageBox.information(self, '训练完成', msg)
            self.init_model()
        else:
            self.train_log.append(f'\n❌ {msg}')
            QMessageBox.critical(self, '训练失败', msg)

    def browse_pth(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择 PTH 模型', './checkpoints/', 'PTH Files (*.pth)'
        )
        if path:
            self.pth_path_edit.setText(path)

    def browse_onnx(self):
        path, _ = QFileDialog.getSaveFileName(
            self, '保存 ONNX 模型', './checkpoints/', 'ONNX Files (*.onnx)'
        )
        if path:
            self.onnx_path_edit.setText(path)

    def convert_model(self):
        pth_path = self.pth_path_edit.text()
        onnx_path = self.onnx_path_edit.text()
        loadsize = self.convert_loadsize.value()

        if not os.path.exists(pth_path):
            QMessageBox.warning(self, '错误', 'PTH 模型文件不存在')
            return

        self.convert_btn.setEnabled(False)
        self.convert_log.clear()

        self.convert_thread = ConvertThread(pth_path, onnx_path, loadsize)
        self.convert_thread.log_signal.connect(self.convert_log.append)
        self.convert_thread.finished_signal.connect(self.on_convert_finished)
        self.convert_thread.start()

    def on_convert_finished(self, success, msg):
        self.convert_btn.setEnabled(True)
        if success:
            self.convert_log.append(f'✅ {msg}')
            QMessageBox.information(self, '转换完成', msg)
        else:
            self.convert_log.append(f'❌ {msg}')
            QMessageBox.critical(self, '转换失败', msg)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())