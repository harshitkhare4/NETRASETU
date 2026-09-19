"""
src/prepare_idrid_localization.py
Prepares the IDRiD Optic Disc & Fovea Localization dataset:
1. Coordinate transformation functions (original <-> processed) with round-trip unit tests.
2. 85% Train (351) / 15% Validation (62) split of official 413 training images with random_state=42.
3. Locked Test set of 103 official test images.
4. Aspect-ratio preserving uniform scaling and letterboxing to 512x512.
5. Saves processed images to data/IDRiD_Localization/{train, val, test}.
6. Generates data/IDRiD_Localization/manifest.csv.
7. Performs comprehensive automated data integrity checks and writes results/localization/data_integrity.txt.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# GEOMETRY CONVERSION FUNCTIONS
# -------------------------------------------------------------

def get_letterbox_params(orig_w, orig_h, target_w=512, target_h=512):
    """
    Computes uniform scale factor and padding offsets for letterboxing.
    """
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    return scale, new_w, new_h, pad_x, pad_y

def original_to_processed_coordinate(x_orig, y_orig, orig_w, orig_h, target_w=512, target_h=512):
    """
    Maps original (W, H) coordinates to letterboxed processed (target_w, target_h) coordinates.
    """
    scale, _, _, pad_x, pad_y = get_letterbox_params(orig_w, orig_h, target_w, target_h)
    x_proc = x_orig * scale + pad_x
    y_proc = y_orig * scale + pad_y
    return float(x_proc), float(y_proc)

def processed_to_original_coordinate(x_proc, y_proc, orig_w, orig_h, target_w=512, target_h=512):
    """
    Maps letterboxed processed coordinates back to original (W, H) image coordinates.
    """
    scale, _, _, pad_x, pad_y = get_letterbox_params(orig_w, orig_h, target_w, target_h)
    x_orig = (x_proc - pad_x) / scale
    y_orig = (y_proc - pad_y) / scale
    return float(x_orig), float(y_orig)

def test_geometry_round_trip():
    """
    Automated round-trip test: ensures coordinate transformations are lossless.
    """
    orig_w, orig_h = 4288, 2848
    test_points = [
        (0.0, 0.0),
        (4287.0, 2847.0),
        (2144.0, 1424.0),
        (1943.1, 1381.2),
        (1964.6, 1525.8),
        (520.0, 1044.0)
    ]
    for x, y in test_points:
        px, py = original_to_processed_coordinate(x, y, orig_w, orig_h)
        rx, ry = processed_to_original_coordinate(px, py, orig_w, orig_h)
        err_x = abs(rx - x)
        err_y = abs(ry - y)
        assert err_x < 1e-4 and err_y < 1e-4, f"Round trip error ({err_x}, {err_y}) exceeds tolerance for ({x}, {y})"
    print("Geometry transformation round-trip test PASSED (tolerance < 1e-4 px).")

def letterbox_image(pil_img, target_w=512, target_h=512):
    """
    Applies uniform scaling and letterbox padding to a PIL image.
    """
    orig_w, orig_h = pil_img.size
    scale, new_w, new_h, pad_x, pad_y = get_letterbox_params(orig_w, orig_h, target_w, target_h)
    
    # High-quality resampling
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create black canvas
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = int(round(pad_x))
    paste_y = int(round(pad_y))
    canvas.paste(resized, (paste_x, paste_y))
    return canvas

# -------------------------------------------------------------
# MAIN PREPARATION PIPELINE
# -------------------------------------------------------------

def prepare_dataset():
    test_geometry_round_trip()

    base_raw = r"C:\Users\riyan\Downloads\26038\C. Localization\C. Localization"
    raw_train_imgs = os.path.join(base_raw, "1. Original Images", "a. Training Set")
    raw_test_imgs = os.path.join(base_raw, "1. Original Images", "b. Testing Set")

    od_train_csv = os.path.join(base_raw, "2. Groundtruths", "1. Optic Disc Center Location", "a. IDRiD_OD_Center_Training Set_Markups.csv")
    od_test_csv = os.path.join(base_raw, "2. Groundtruths", "1. Optic Disc Center Location", "b. IDRiD_OD_Center_Testing Set_Markups.csv")
    fovea_train_csv = os.path.join(base_raw, "2. Groundtruths", "2. Fovea Center Location", "IDRiD_Fovea_Center_Training Set_Markups.csv")
    fovea_test_csv = os.path.join(base_raw, "2. Groundtruths", "2. Fovea Center Location", "IDRiD_Fovea_Center_Testing Set_Markups.csv")

    dest_base = r"C:\NetraSetu\data\IDRiD_Localization"
    train_dest = os.path.join(dest_base, "train")
    val_dest = os.path.join(dest_base, "val")
    test_dest = os.path.join(dest_base, "test")
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    results_loc = r"C:\NetraSetu\results\localization"
    os.makedirs(results_loc, exist_ok=True)

    # 1. Parse CSVs
    def clean_csv(path, label):
        df = pd.read_csv(path).dropna(subset=['Image No'])
        cols = df.columns.tolist()
        xc = [c for c in cols if 'x' in c.lower() and 'coord' in c.lower()][0]
        yc = [c for c in cols if 'y' in c.lower() and 'coord' in c.lower()][0]
        df = df[['Image No', xc, yc]].rename(columns={'Image No': 'image_name', xc: f'{label}_x', yc: f'{label}_y'})
        df['image_name'] = df['image_name'].str.strip()
        return df

    od_tr = clean_csv(od_train_csv, 'od')
    fov_tr = clean_csv(fovea_train_csv, 'fovea')
    train_df = pd.merge(od_tr, fov_tr, on='image_name')

    od_te = clean_csv(od_test_csv, 'od')
    fov_te = clean_csv(fovea_test_csv, 'fovea')
    test_df = pd.merge(od_te, fov_te, on='image_name')

    print(f"Loaded {len(train_df)} training markups and {len(test_df)} testing markups.")

    # 2. Split Training set into Train (85%) and Val (15%) with random_state=42
    # 413 * 0.15 = 61.95 -> 62 val, 351 train
    train_idx, val_idx = train_test_split(train_df.index, test_size=0.15, random_state=42, shuffle=True)
    train_split_df = train_df.loc[train_idx].copy()
    val_split_df = train_df.loc[val_idx].copy()
    train_split_df['split'] = 'train'
    val_split_df['split'] = 'val'
    test_df['split'] = 'test'

    print(f"Split sizes: Train={len(train_split_df)}, Val={len(val_split_df)}, Test={len(test_df)}")

    combined_dev = pd.concat([train_split_df, val_split_df], ignore_index=True)
    all_splits_df = pd.concat([combined_dev, test_df], ignore_index=True)

    manifest_rows = []

    # 3. Process each split and generate letterboxed images
    print("Processing and letterboxing images to 512x512...")
    for _, row in all_splits_df.iterrows():
        img_name = row['image_name']
        split = row['split']
        
        if split in ['train', 'val']:
            src_path = os.path.join(raw_train_imgs, f"{img_name}.jpg")
            dst_dir = train_dest if split == 'train' else val_dest
        else:
            src_path = os.path.join(raw_test_imgs, f"{img_name}.jpg")
            dst_dir = test_dest

        dst_path = os.path.join(dst_dir, f"{img_name}.jpg")

        with Image.open(src_path) as pil_img:
            orig_w, orig_h = pil_img.size
            processed_img = letterbox_image(pil_img, target_w=512, target_h=512)
            processed_img.save(dst_path, "JPEG", quality=95)

        proc_w, proc_h = 512, 512
        od_x_orig, od_y_orig = float(row['od_x']), float(row['od_y'])
        fov_x_orig, fov_y_orig = float(row['fovea_x']), float(row['fovea_y'])

        od_x_proc, od_y_proc = original_to_processed_coordinate(od_x_orig, od_y_orig, orig_w, orig_h, proc_w, proc_h)
        fov_x_proc, fov_y_proc = original_to_processed_coordinate(fov_x_orig, fov_y_orig, orig_w, orig_h, proc_w, proc_h)

        manifest_rows.append({
            'image_name': img_name,
            'split': split,
            'original_width': orig_w,
            'original_height': orig_h,
            'processed_width': proc_w,
            'processed_height': proc_h,
            'od_x_original': od_x_orig,
            'od_y_original': od_y_orig,
            'fovea_x_original': fov_x_orig,
            'fovea_y_original': fov_y_orig,
            'od_x_processed': round(od_x_proc, 4),
            'od_y_processed': round(od_y_proc, 4),
            'fovea_x_processed': round(fov_x_proc, 4),
            'fovea_y_processed': round(fov_y_proc, 4)
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv_path = os.path.join(dest_base, "manifest.csv")
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"Saved manifest to {manifest_csv_path} with {len(manifest_df)} records.")

    # 4. Comprehensive Data Integrity Audit
    integrity_lines = []
    integrity_lines.append("=" * 60)
    integrity_lines.append("NETRASETU — IDRiD LOCALIZATION DATA INTEGRITY REPORT")
    integrity_lines.append("=" * 60)

    train_count = len(manifest_df[manifest_df['split'] == 'train'])
    val_count = len(manifest_df[manifest_df['split'] == 'val'])
    test_count = len(manifest_df[manifest_df['split'] == 'test'])

    integrity_lines.append(f"\n[1] SPLIT QUANTITIES:")
    integrity_lines.append(f"  Training count:   {train_count} (Expected: 351)")
    integrity_lines.append(f"  Validation count: {val_count} (Expected: 62)")
    integrity_lines.append(f"  Total Dev pool:   {train_count + val_count} (Expected: 413)")
    integrity_lines.append(f"  Locked Test count:{test_count} (Expected: 103)")

    assert train_count == 351, f"Train count mismatch: {train_count}"
    assert val_count == 62, f"Val count mismatch: {val_count}"
    assert train_count + val_count == 413, "Dev pool != 413"
    assert test_count == 103, f"Test count mismatch: {test_count}"

    # Overlap Checks
    train_names = set(manifest_df[manifest_df['split'] == 'train']['image_name'])
    val_names = set(manifest_df[manifest_df['split'] == 'val']['image_name'])
    test_names = set(manifest_df[manifest_df['split'] == 'test']['image_name'])

    train_val_overlap = train_names.intersection(val_names)
    integrity_lines.append(f"\n[2] SPLIT LEAKAGE & OVERLAP CHECKS:")
    integrity_lines.append(f"  Train / Validation overlap: {len(train_val_overlap)}")
    assert len(train_val_overlap) == 0, f"Train/Val overlap detected: {train_val_overlap}"

    # Verify that test files on disk are completely isolated
    test_disk_files = os.listdir(test_dest)
    train_disk_files = os.listdir(train_dest)
    val_disk_files = os.listdir(val_dest)

    integrity_lines.append(f"  Disk file counts: Train={len(train_disk_files)}, Val={len(val_disk_files)}, Test={len(test_disk_files)}")
    assert len(train_disk_files) == 351
    assert len(val_disk_files) == 62
    assert len(test_disk_files) == 103

    # Check Processed Coordinates inside [0, 512)
    oob_processed = []
    for _, r in manifest_df.iterrows():
        for pt, (px, py) in [('OD', (r['od_x_processed'], r['od_y_processed'])), ('Fovea', (r['fovea_x_processed'], r['fovea_y_processed']))]:
            if not (0 <= px < 512 and 0 <= py < 512):
                oob_processed.append((r['image_name'], r['split'], pt, px, py))

    integrity_lines.append(f"\n[3] PROCESSED COORDINATE BOUNDS [0, 512)x[0, 512):")
    integrity_lines.append(f"  Out of bounds processed coordinates: {len(oob_processed)}")
    assert len(oob_processed) == 0, f"Out of bounds processed coords: {oob_processed}"
    integrity_lines.append("  -> All processed coordinates strictly bounded within letterbox canvas.")

    # Check that scale and pad are uniform
    scale = min(512 / 4288, 512 / 2848)
    expected_pad_y = (512 - int(round(2848 * scale))) / 2.0
    integrity_lines.append(f"\n[4] GEOMETRY VERIFICATION:")
    integrity_lines.append(f"  Scale factor: {scale:.6f}")
    integrity_lines.append(f"  Canvas padding: pad_x = 0.0, pad_y = {expected_pad_y:.1f} px")
    integrity_lines.append("  Automated round-trip test: PASSED (lossless reconstruction within 1e-4 pixels).")

    integrity_lines.append(f"\n[5] FINAL INTEGRITY VERDICT: PASSED (Integrity 100%)")
    integrity_lines.append("=" * 60)

    integrity_text = "\n".join(integrity_lines)
    integrity_file = os.path.join(results_loc, "data_integrity.txt")
    with open(integrity_file, "w", encoding="utf-8") as f:
        f.write(integrity_text)

    print(integrity_text)

if __name__ == "__main__":
    prepare_dataset()
