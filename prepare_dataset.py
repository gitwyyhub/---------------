import os
import glob
import shutil
import xml.etree.ElementTree as ET
from PIL import Image


def crop_and_save(src_dir, dst_dir, class_names):
    os.makedirs(dst_dir, exist_ok=True)
    for name in class_names:
        os.makedirs(os.path.join(dst_dir, name), exist_ok=True)

    xml_files = glob.glob(os.path.join(src_dir, '*.xml'))
    count = {name: 0 for name in class_names}

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename')
        if filename is None:
            continue
        img_path = os.path.join(src_dir, filename.text)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
        except Exception:
            continue

        for obj in root.findall('object'):
            name_el = obj.find('name')
            if name_el is None:
                continue
            cls_name = name_el.text
            if cls_name not in class_names:
                continue

            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            try:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
            except Exception:
                continue

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img.width, xmax)
            ymax = min(img.height, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue

            cropped = img.crop((xmin, ymin, xmax, ymax))
            save_name = f'{os.path.splitext(filename.text)[0]}_{count[cls_name]}.jpg'
            save_path = os.path.join(dst_dir, cls_name, save_name)
            cropped.save(save_path, 'JPEG')
            count[cls_name] += 1

    return count


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(base_dir, 'datasets')
    class_names = ['WBC', 'RBC', 'Platelets']

    for split in ['train', 'test', 'val']:
        src_dir = os.path.join(datasets_dir, split)
        if not os.path.exists(src_dir):
            print(f'[SKIP] {src_dir} 不存在')
            continue

        dst_dir = os.path.join(datasets_dir, split + '_classified')
        print(f'[处理] {split}: {src_dir} -> {dst_dir}')
        count = crop_and_save(src_dir, dst_dir, class_names)

        for name, num in count.items():
            print(f'  {name}: {num} 张')
        total = sum(count.values())
        print(f'  合计: {total} 张\n')


if __name__ == '__main__':
    main()