import os
import shutil
import csv
import random
from collections import defaultdict

BASE = r"g:\wyy\python\白细胞、红细胞、血小板识别系统\datasets"
OLD_PKG = os.path.join(BASE, "image_tabular_datasets", "package")

random.seed(42)

print("=" * 60)
print("开始整理 datasets 文件夹...")
print("=" * 60)

# ============================================================
# 1. BCCD 血细胞检测数据集
# ============================================================
print("\n[1/3] 整理 BCCD 血细胞检测数据集...")

BCCD_SRC = os.path.join(OLD_PKG, "BCCD_blood_cell")
BCCD_DST = os.path.join(BASE, "BCCD")
TEST_SRC = os.path.join(BASE, "test")
CLASSIFIED_SRC = os.path.join(BASE, "test_classified")

os.makedirs(BCCD_DST, exist_ok=True)

# 读取 bccd_annotations.csv 获取所有图像列表
csv_path = os.path.join(BCCD_SRC, "bccd_annotations.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_bccd_images = [row["image"] for row in reader]

print(f"    BCCD CSV 中共 {len(all_bccd_images)} 张图像")

# 从现有 test/ 目录获取测试集图像编号
existing_test = set()
if os.path.exists(TEST_SRC):
    for fname in os.listdir(TEST_SRC):
        if fname.endswith(".jpg"):
            base_name = fname.split("_jpg")[0] + ".jpg"
            existing_test.add(base_name)

print(f"    已有测试集图像: {len(existing_test)} 张")

# 排除测试集，剩余作为训练+验证
remaining = [img for img in all_bccd_images if img not in existing_test]
random.shuffle(remaining)

n_train = int(len(remaining) * 0.7)
train_images = set(remaining[:n_train])
val_images = set(remaining[n_train:])

print(f"    训练集: {len(train_images)} 张")
print(f"    验证集: {len(val_images)} 张")
print(f"    测试集: {len(existing_test)} 张")

# 创建 BCCD 目录结构
for split, img_set in [("train", train_images), ("val", val_images), ("test", existing_test)]:
    img_dir = os.path.join(BCCD_DST, split, "images")
    ann_dir = os.path.join(BCCD_DST, split, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

# 移动 train/val 图像和标注
for split, img_set in [("train", train_images), ("val", val_images)]:
    src_img_dir = os.path.join(BCCD_SRC, "JPEGImages")
    src_ann_dir = os.path.join(BCCD_SRC, "Annotations")
    for img_name in img_set:
        src_img = os.path.join(src_img_dir, img_name)
        ann_name = img_name.replace(".jpg", ".xml")
        src_ann = os.path.join(src_ann_dir, ann_name)
        dst_img = os.path.join(BCCD_DST, split, "images", img_name)
        dst_ann = os.path.join(BCCD_DST, split, "annotations", ann_name)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_ann):
            shutil.move(src_ann, dst_ann)

# 移动 test 图像和标注（从 datasets/test/ 和 BCCD Annotations）
for img_name in existing_test:
    base_num = img_name.replace("BloodImage_", "").replace(".jpg", "")
    # 匹配 datasets/test/ 中以 BloodImage_XXXXX 开头的文件
    for fname in os.listdir(TEST_SRC):
        if f"BloodImage_{base_num}" in fname:
            src_file = os.path.join(TEST_SRC, fname)
            if fname.endswith(".jpg"):
                dst_file = os.path.join(BCCD_DST, "test", "images", img_name)
            elif fname.endswith(".xml"):
                ann_name = img_name.replace(".jpg", ".xml")
                dst_file = os.path.join(BCCD_DST, "test", "annotations", ann_name)
            else:
                continue
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                shutil.move(src_file, dst_file)
    ann_name = img_name.replace(".jpg", ".xml")
    src_ann = os.path.join(BCCD_SRC, "Annotations", ann_name)
    dst_ann = os.path.join(BCCD_DST, "test", "annotations", ann_name)
    if os.path.exists(src_ann):
        shutil.move(src_ann, dst_ann)

# 复制 CSV 文件
shutil.copy2(csv_path, os.path.join(BCCD_DST, "bccd_annotations.csv"))

# 移动 test_classified 到 BCCD 下
if os.path.exists(CLASSIFIED_SRC):
    classified_dst = os.path.join(BCCD_DST, "classified_crops")
    if os.path.exists(classified_dst):
        shutil.rmtree(classified_dst)
    shutil.move(CLASSIFIED_SRC, classified_dst)
    print(f"    已移动 classified_crops 到 BCCD/")

# 清理空的 test/ 目录
if os.path.exists(TEST_SRC) and not os.listdir(TEST_SRC):
    os.rmdir(TEST_SRC)

print("    BCCD 整理完成!")

# ============================================================
# 2. BloodMNIST 血细胞分类数据集
# ============================================================
print("\n[2/3] 整理 BloodMNIST 血细胞分类数据集...")

BLOODMNIST_SRC = os.path.join(OLD_PKG, "BloodMNIST_blood_cell")
BLOODMNIST_DST = os.path.join(BASE, "BloodMNIST")

os.makedirs(BLOODMNIST_DST, exist_ok=True)

# 读取标签 CSV
labels_csv = os.path.join(BLOODMNIST_SRC, "bloodmnist_labels.csv")
with open(labels_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    labels_data = list(reader)

print(f"    BloodMNIST 标签共 {len(labels_data)} 条")

# 创建 train/val/test 目录
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BLOODMNIST_DST, split), exist_ok=True)

# 复制 CSV 和 sample_images
shutil.copy2(labels_csv, os.path.join(BLOODMNIST_DST, "bloodmnist_labels.csv"))
sample_src = os.path.join(BLOODMNIST_SRC, "sample_images")
if os.path.exists(sample_src):
    sample_dst = os.path.join(BLOODMNIST_DST, "sample_images")
    if os.path.exists(sample_dst):
        shutil.rmtree(sample_dst)
    shutil.move(sample_src, sample_dst)

print("    BloodMNIST 整理完成!")

# ============================================================
# 3. BrainTumor MRI 脑肿瘤数据集
# ============================================================
print("\n[3/3] 整理 BrainTumor MRI 脑肿瘤数据集...")

BRAIN_SRC = os.path.join(OLD_PKG, "BrainTumor_MRI_Cheng")
BRAIN_DST = os.path.join(BASE, "BrainTumor")

os.makedirs(BRAIN_DST, exist_ok=True)

# 读取 CSV，按 patient_id 分组
tumor_csv = os.path.join(BRAIN_SRC, "braintumor_labels.csv")
patient_groups = defaultdict(list)
tumor_data = []
with open(tumor_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["patient_id"]
        patient_groups[pid].append(row)
        tumor_data.append(row)

all_patients = list(patient_groups.keys())
random.shuffle(all_patients)

n_train_p = int(len(all_patients) * 0.7)
n_val_p = int(len(all_patients) * 0.15)

train_patients = set(all_patients[:n_train_p])
val_patients = set(all_patients[n_train_p:n_train_p + n_val_p])
test_patients = set(all_patients[n_train_p + n_val_p:])

print(f"    总患者数: {len(all_patients)}")
print(f"    训练集患者: {len(train_patients)}")
print(f"    验证集患者: {len(val_patients)}")
print(f"    测试集患者: {len(test_patients)}")

patient_to_split = {}
for pid in train_patients:
    patient_to_split[pid] = "train"
for pid in val_patients:
    patient_to_split[pid] = "val"
for pid in test_patients:
    patient_to_split[pid] = "test"

# 创建目录结构
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BRAIN_DST, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(BRAIN_DST, split, "masks"), exist_ok=True)

src_images_dir = os.path.join(BRAIN_SRC, "images")
src_masks_dir = os.path.join(BRAIN_SRC, "masks")

# 移动图像和掩码
for row in tumor_data:
    pid = row["patient_id"]
    split = patient_to_split[pid]
    img_name = os.path.basename(row["image"])
    mask_name = os.path.basename(row["mask"])

    src_img = os.path.join(src_images_dir, img_name)
    src_mask = os.path.join(src_masks_dir, mask_name)

    dst_img = os.path.join(BRAIN_DST, split, "images", img_name)
    dst_mask = os.path.join(BRAIN_DST, split, "masks", mask_name)

    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
    if os.path.exists(src_mask):
        shutil.move(src_mask, dst_mask)

# 复制 CSV
shutil.copy2(tumor_csv, os.path.join(BRAIN_DST, "braintumor_labels.csv"))

print("    BrainTumor 整理完成!")

# ============================================================
# 清理旧的空目录
# ============================================================
print("\n[清理] 删除旧的空目录...")

old_bccd = os.path.join(OLD_PKG, "BCCD_blood_cell")
old_bloodmnist = os.path.join(OLD_PKG, "BloodMNIST_blood_cell")
old_brain = os.path.join(OLD_PKG, "BrainTumor_MRI_Cheng")

for old_dir in [old_bccd, old_bloodmnist, old_brain]:
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir, ignore_errors=True)

old_pkg_dir = os.path.join(OLD_PKG)
if os.path.exists(old_pkg_dir):
    try:
        os.rmdir(old_pkg_dir)
    except OSError:
        pass

old_image_tabular = os.path.join(BASE, "image_tabular_datasets")
if os.path.exists(old_image_tabular):
    try:
        os.rmdir(old_image_tabular)
    except OSError:
        pass

print("\n" + "=" * 60)
print("所有数据集整理完成!")
print("=" * 60)
print("""
新的目录结构:
  datasets/
    BCCD/
      train/images/   + annotations/
      val/images/     + annotations/
      test/images/    + annotations/
      classified_crops/
      bccd_annotations.csv
    BloodMNIST/
      train/  val/  test/
      sample_images/
      bloodmnist_labels.csv
    BrainTumor/
      train/images/   + masks/
      val/images/     + masks/
      test/images/    + masks/
      braintumor_labels.csv
""")