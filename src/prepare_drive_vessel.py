"""
src/prepare_drive_vessel.py
Extracts and prepares the official DRIVE retinal vessel segmentation dataset from datasets.zip.
- Training pool (20 images, 21..40): has official manual vessel ground truth.
  Split: 85% Train (17 images) / 15% Validation (3 images) with random_state=42.
- Testing set (20 images, 01..20): locked test set with images and FOV masks.
- All images and masks letterboxed/resized to 512x512 PNG.
- Manifest saved to data/DRIVE/manifest.csv.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def prepare_drive():
    print("=" * 60)
    print("NETRASETU — DRIVE RETINAL VESSEL DATASET PREPARATION")
    print("=" * 60)

    zip_path = r"C:\Users\riyan\Downloads\26038\datasets.zip"
    dest_base = r"C:\NetraSetu\data\DRIVE"

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_base, split, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(dest_base, split, 'fov_masks'), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as root_zip:
        training_zip_data = root_zip.read('training.zip')
        test_zip_data = root_zip.read('test.zip')

    train_records = []
    with zipfile.ZipFile(io.BytesIO(training_zip_data)) as tz:
        train_img_names = sorted([f for f in tz.namelist() if f.startswith('training/images/') and f.endswith(('.tif', '.png', '.jpg'))])
        print(f"Found {len(train_img_names)} official training images in training.zip (IDs 21-40)")

        for img_entry in train_img_names:
            base_name = os.path.basename(img_entry)
            num_id = base_name.split('_')[0]
            mask_entry = f"training/1st_manual/{num_id}_manual1.gif"
            fov_entry = f"training/mask/{num_id}_training_mask.gif"

            img_bytes = tz.read(img_entry)
            mask_bytes = tz.read(mask_entry)
            fov_bytes = tz.read(fov_entry)

            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            mask_pil = Image.open(io.BytesIO(mask_bytes)).convert('L')
            fov_pil = Image.open(io.BytesIO(fov_bytes)).convert('L')

            train_records.append({
                'id': num_id,
                'img_pil': img_pil,
                'mask_pil': mask_pil,
                'fov_pil': fov_pil,
                'orig_size': img_pil.size
            })

    test_records = []
    with zipfile.ZipFile(io.BytesIO(test_zip_data)) as tz:
        test_img_names = sorted([f for f in tz.namelist() if f.startswith('test/images/') and f.endswith(('.tif', '.png', '.jpg'))])
        print(f"Found {len(test_img_names)} official test images in test.zip (IDs 01-20)")

        for img_entry in test_img_names:
            base_name = os.path.basename(img_entry)
            num_id = base_name.split('_')[0]
            fov_entry = f"test/mask/{num_id}_test_mask.gif"

            img_bytes = tz.read(img_entry)
            fov_bytes = tz.read(fov_entry)

            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            fov_pil = Image.open(io.BytesIO(fov_bytes)).convert('L')

            test_records.append({
                'id': num_id,
                'img_pil': img_pil,
                'mask_pil': None, # Official test ground truths were withheld in DRIVE distribution
                'fov_pil': fov_pil,
                'orig_size': img_pil.size
            })

    # Split 20 training images: 85% train (17) / 15% val (3), random_state=42
    indices = list(range(len(train_records)))
    tr_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
    print(f"Development Split: Train = {len(tr_idx)} images, Validation = {len(val_idx)} images")
    print(f"Locked Test Set:   {len(test_records)} images")

    manifest_rows = []

    def save_set(records_subset, split_name):
        for rec in records_subset:
            img_id = rec['id']
            img = rec['img_pil'].resize((512, 512), Image.Resampling.LANCZOS)
            fov = rec['fov_pil'].resize((512, 512), Image.Resampling.NEAREST)

            img_save_path = os.path.join(dest_base, split_name, 'images', f"drive_{img_id}.png")
            fov_save_path = os.path.join(dest_base, split_name, 'fov_masks', f"drive_{img_id}.png")
            img.save(img_save_path, "PNG")
            fov.save(fov_save_path, "PNG")

            has_mask = (rec['mask_pil'] is not None)
            if has_mask:
                mask = rec['mask_pil'].resize((512, 512), Image.Resampling.NEAREST)
                mask_save_path = os.path.join(dest_base, split_name, 'masks', f"drive_{img_id}.png")
                mask.save(mask_save_path, "PNG")

            manifest_rows.append({
                'image_id': f"drive_{img_id}",
                'split': split_name,
                'orig_width': rec['orig_size'][0],
                'orig_height': rec['orig_size'][1],
                'proc_width': 512,
                'proc_height': 512,
                'has_vessel_mask': has_mask
            })

    save_set([train_records[i] for i in tr_idx], 'train')
    save_set([train_records[i] for i in val_idx], 'val')
    save_set(test_records, 'test')

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(dest_base, 'manifest.csv')
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Saved manifest to {manifest_csv} with {len(manifest_df)} images.")
    print("DRIVE preparation complete!")

if __name__ == '__main__':
    prepare_drive()
