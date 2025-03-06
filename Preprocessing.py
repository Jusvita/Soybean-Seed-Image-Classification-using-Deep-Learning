import os
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from collections import Counter
from sklearn.utils import resample
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Connecting to the Original Dataset
dataset_dir = '/content/drive/MyDrive/..../Dataset-Original'
classes = ['Broken soybeans', 'Immature soybeans', 'Intact soybeans', 'Skin-damaged soybeans', 'Spotted soybeans']

# Undersampling Dataset
target_samples = 1000
parent_dir = '/content/drive/MyDrive/..../Undersampled'
remaining_dir = '/content/drive/MyDrive/...../Remaining Dataset'

# Looping untuk setiap kelas
for c in classes:
    class_dir = os.path.join(dataset_dir, c)
    images = os.listdir(class_dir)

    # Memfilter hanya file gambar, jika ada folder di dalamnya
    images = [img for img in images if os.path.isfile(os.path.join(class_dir, img))]

    # Melakukan undersampling jika jumlah gambar lebih dari target_samples
    if len(images) > target_samples:
        # Resample gambar untuk undersampling
        undersampled_images = resample(images, replace=False, n_samples=target_samples, random_state=42)

        # Menyimpan gambar yang di-undersample
        undersampled_dir = os.path.join(parent_dir, c)
        os.makedirs(undersampled_dir, exist_ok=True)

        for img in undersampled_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(undersampled_dir, img)
            shutil.copy(src, dst)

        # Menyimpan gambar yang tidak di-undersample ke folder sisa
        remaining_images = list(set(images) - set(undersampled_images))
        remaining_class_dir = os.path.join(remaining_dir, c)
        os.makedirs(remaining_class_dir, exist_ok=True)

        for img in remaining_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(remaining_class_dir, img)
            shutil.copy(src, dst)

    else:
        # Jika jumlah gambar di kelas sudah lebih kecil atau sama dengan target_samples, salin semua gambar
        undersampled_dir = os.path.join(parent_dir, c)
        os.makedirs(undersampled_dir, exist_ok=True)
        for img in images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(undersampled_dir, img)
            shutil.copy(src, dst)

#Splitting Train-val & Testing
# Proporsi pembagian
train_val_split = 0.8

# Statistik hasil pembagian
stats = {}

# Loop setiap kelas dalam dataset
for class_name in os.listdir(parent_dir):
    class_path = os.path.join(parent_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Membuat folder kelas di dataset_kedelai dan data_test
    os.makedirs(os.path.join(dataset_kedelai_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(data_test_dir, class_name), exist_ok=True)

    # Mendapatkan daftar semua file gambar di kelas ini
    all_files = os.listdir(class_path)
    random.shuffle(all_files)

    # Membagi file ke train_val dan test
    split_idx = int(len(all_files) * train_val_split)
    train_val_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    # Memindahkan file ke folder yang sesuai
    for file_name in train_val_files:
        shutil.copy(
            os.path.join(class_path, file_name),
            os.path.join(dataset_kedelai_dir, class_name, file_name)
        )
    for file_name in test_files:
        shutil.copy(
            os.path.join(class_path, file_name),
            os.path.join(data_test_dir, class_name, file_name)
        )

#Dataset Augmentation
augmentations = {
    'brightness': transforms.ColorJitter(brightness=0.3),
    'contrast': transforms.ColorJitter(contrast=0.2),
    'saturation': transforms.ColorJitter(saturation=0.3),
    'hue': transforms.ColorJitter(hue=0.1)
}
# Iterasi setiap folder kelas di dalam dataset asli
for class_folder in os.listdir(dataset_kedelai_dir):
    class_path = os.path.join(dataset_kedelai_dir, class_folder)
    if os.path.isdir(class_path):
        for image_name in tqdm(os.listdir(class_path), desc=f'Processing {class_folder}'):
            image_path = os.path.join(class_path, image_name)

            # Buka gambar dengan PIL
            image = Image.open(image_path)

            # Lakukan setiap augmentasi secara terpisah
            for aug_type, aug_transform in augmentations.items():
                # Terapkan transformasi augmentasi
                augmented_image = aug_transform(image)

# Splitting Dataset before Augmentation (Train:Val for 70:30, 80:20, 90:10)
def split_and_copy_dataset(base_dir, train_dir, val_dir, val_size, random_state=42):
    # Membuat folder train dan val untuk setiap kelas
    classes = os.listdir(base_dir)

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Proses setiap kelas
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        images = os.listdir(cls_dir)

        # Bagi gambar menjadi train dan val (tanpa test)
        train_images, val_images = train_test_split(images, test_size=val_size, random_state=random_state)

        # Salin gambar ke folder yang sesuai
        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))

# Combining augmented Dataset and Original Dataset
# Function to copy images from a source directory to the output directory
def copy_images(src_dir, dest_dir, cls):
    class_path = os.path.join(src_dir, cls)
    if os.path.exists(class_path):
        for img_file in os.listdir(class_path):
            img_src_path = os.path.join(class_path, img_file)
            img_dest_path = os.path.join(dest_dir, cls, img_file)
            shutil.copy(img_src_path, img_dest_path)

# Copy images from augmentation directories
for augmentation in augmentations:
    augmentation_path = os.path.join(augment_base_dir, augmentation)
    for cls in classes:
        copy_images(augmentation_path, output_dir, cls)

# Copy images from undersampled dataset
for cls in classes:
    copy_images(undersample_base_dir, output_dir, cls)

# Splitting Dataset after Augmentation (Train:Val for 70:30, 80:20 and 90:10)
def split_and_copy_dataset(base_dir, train_dir, val_dir, val_size, random_state=42):
    # Membuat folder train dan val untuk setiap kelas
    classes = os.listdir(base_dir)

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Proses setiap kelas
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        images = os.listdir(cls_dir)

        # Bagi gambar menjadi train dan val (tanpa test)
        train_images, val_images = train_test_split(images, test_size=val_size, random_state=random_state)

        # Salin gambar ke folder yang sesuai
        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))

