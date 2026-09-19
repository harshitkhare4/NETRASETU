"""
src/verify_idrid_localization.py
Verification script for IDRiD C. Localization Dataset (Optic Disc & Fovea Center Coordinates).

Verifies:
- Exactly 413 training images and 103 testing images
- Every training image has OD and Fovea coordinates
- Every testing image has OD and Fovea coordinates
- Coordinates within bounds: 0 <= x < width, 0 <= y < height
- No missing annotations, duplicate image names, or missing image files
- Image dimension consistency

Saves report to results/localization/verification.txt.
"""

import os
import sys
import pandas as pd
from PIL import Image

def verify_dataset():
    base_dir = r"C:\Users\riyan\Downloads\26038\C. Localization\C. Localization"
    img_train_dir = os.path.join(base_dir, "1. Original Images", "a. Training Set")
    img_test_dir = os.path.join(base_dir, "1. Original Images", "b. Testing Set")
    
    od_train_csv = os.path.join(base_dir, "2. Groundtruths", "1. Optic Disc Center Location", "a. IDRiD_OD_Center_Training Set_Markups.csv")
    od_test_csv = os.path.join(base_dir, "2. Groundtruths", "1. Optic Disc Center Location", "b. IDRiD_OD_Center_Testing Set_Markups.csv")
    fovea_train_csv = os.path.join(base_dir, "2. Groundtruths", "2. Fovea Center Location", "IDRiD_Fovea_Center_Training Set_Markups.csv")
    fovea_test_csv = os.path.join(base_dir, "2. Groundtruths", "2. Fovea Center Location", "IDRiD_Fovea_Center_Testing Set_Markups.csv")

    out_dir = r"C:\NetraSetu\results\localization"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "verification.txt")

    lines = []
    lines.append("=" * 60)
    lines.append("NETRASETU — IDRiD LOCALIZATION DATASET VERIFICATION AUDIT")
    lines.append("=" * 60)

    # 1. Check Image Files on Disk
    train_files = [f for f in os.listdir(img_train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
    test_files = [f for f in os.listdir(img_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
    
    lines.append(f"\n[1] IMAGE FILE COUNTS ON DISK:")
    lines.append(f"  Training images found: {len(train_files)} (Expected: 413)")
    lines.append(f"  Testing images found:  {len(test_files)} (Expected: 103)")
    assert len(train_files) == 413, f"Expected 413 training images, found {len(train_files)}"
    assert len(test_files) == 103, f"Expected 103 testing images, found {len(test_files)}"
    lines.append("  -> Image counts on disk match exact specification.")

    # Check for duplicate names within splits and between splits
    train_stems = [os.path.splitext(f)[0] for f in train_files]
    test_stems = [os.path.splitext(f)[0] for f in test_files]
    lines.append(f"\n[2] IMAGE NAME DUPLICATE CHECK:")
    lines.append(f"  Unique training image stems: {len(set(train_stems))} / {len(train_stems)}")
    lines.append(f"  Unique testing image stems:  {len(set(test_stems))} / {len(test_stems)}")
    assert len(set(train_stems)) == len(train_stems), "Duplicate training image stems detected!"
    assert len(set(test_stems)) == len(test_stems), "Duplicate testing image stems detected!"

    # Note on IDs: Both training and testing use IDRiD_001.jpg etc. but live in separate directories and splits!
    lines.append(f"  Note: Train and test sets both use official IDRiD numbering (IDRiD_001..IDRiD_413 and IDRiD_001..IDRiD_103), strictly segregated in respective folders.")

    # 2. Check Dimensions
    lines.append(f"\n[3] IMAGE DIMENSION AUDIT:")
    dim_counts_train = {}
    for f in train_files:
        with Image.open(os.path.join(img_train_dir, f)) as img:
            size = img.size
            dim_counts_train[size] = dim_counts_train.get(size, 0) + 1

    dim_counts_test = {}
    for f in test_files:
        with Image.open(os.path.join(img_test_dir, f)) as img:
            size = img.size
            dim_counts_test[size] = dim_counts_test.get(size, 0) + 1

    lines.append(f"  Training dimension distribution: {dim_counts_train}")
    lines.append(f"  Testing dimension distribution:  {dim_counts_test}")
    for sz, count in dim_counts_train.items():
        lines.append(f"    Train size (W={sz[0]}, H={sz[1]}): {count} images")
    for sz, count in dim_counts_test.items():
        lines.append(f"    Test size  (W={sz[0]}, H={sz[1]}): {count} images")

    # 3. Read and Clean CSV Annotations
    lines.append(f"\n[4] GROUND TRUTH CSV PARSING & CLEANING:")
    def parse_markup_csv(csv_path, expected_count, label_name):
        df = pd.read_csv(csv_path)
        valid = df.dropna(subset=['Image No']).copy()
        # Ensure correct column names
        cols = valid.columns.tolist()
        x_col = [c for c in cols if 'x' in c.lower() and 'coord' in c.lower()][0]
        y_col = [c for c in cols if 'y' in c.lower() and 'coord' in c.lower()][0]
        valid = valid[['Image No', x_col, y_col]].rename(columns={
            'Image No': 'image_no',
            x_col: f'{label_name}_x',
            y_col: f'{label_name}_y'
        })
        valid['image_no'] = valid['image_no'].str.strip()
        lines.append(f"  {os.path.basename(csv_path)}: {len(valid)} valid entries parsed (Expected: {expected_count})")
        assert len(valid) == expected_count, f"Expected {expected_count} rows in {csv_path}, got {len(valid)}"
        return valid

    df_od_train = parse_markup_csv(od_train_csv, 413, 'od')
    df_fovea_train = parse_markup_csv(fovea_train_csv, 413, 'fovea')
    df_od_test = parse_markup_csv(od_test_csv, 103, 'od')
    df_fovea_test = parse_markup_csv(fovea_test_csv, 103, 'fovea')

    # Merge Training Annotations
    train_annos = pd.merge(df_od_train, df_fovea_train, on='image_no', how='inner')
    lines.append(f"  Merged training annotations: {len(train_annos)} / 413")
    assert len(train_annos) == 413, f"Merged training annotations incomplete: {len(train_annos)} != 413"

    # Merge Testing Annotations
    test_annos = pd.merge(df_od_test, df_fovea_test, on='image_no', how='inner')
    lines.append(f"  Merged testing annotations:  {len(test_annos)} / 103")
    assert len(test_annos) == 103, f"Merged testing annotations incomplete: {len(test_annos)} != 103"

    # 4. Check for missing images in CSV or files missing on disk
    train_disk_stems = set(train_stems)
    test_disk_stems = set(test_stems)
    
    missing_train_files = [im for im in train_annos['image_no'] if im not in train_disk_stems]
    missing_test_files = [im for im in test_annos['image_no'] if im not in test_disk_stems]
    lines.append(f"\n[5] MISSING FILE CHECK:")
    lines.append(f"  Training images in CSV missing on disk: {len(missing_train_files)}")
    lines.append(f"  Testing images in CSV missing on disk:  {len(missing_test_files)}")
    assert len(missing_train_files) == 0, f"Missing train images: {missing_train_files}"
    assert len(missing_test_files) == 0, f"Missing test images: {missing_test_files}"

    # 5. Check Coordinate Bounds: 0 <= x < W, 0 <= y < H
    lines.append(f"\n[6] COORDINATE BOUNDARY VALIDATION:")
    def check_bounds(annos_df, img_dir, split_name):
        out_of_bounds = []
        for _, row in annos_df.iterrows():
            img_name = row['image_no']
            img_path = os.path.join(img_dir, f"{img_name}.jpg")
            with Image.open(img_path) as img:
                w, h = img.size
            od_x, od_y = row['od_x'], row['od_y']
            fovea_x, fovea_y = row['fovea_x'], row['fovea_y']

            if not (0 <= od_x < w and 0 <= od_y < h):
                out_of_bounds.append((img_name, 'OD', od_x, od_y, w, h))
            if not (0 <= fovea_x < w and 0 <= fovea_y < h):
                out_of_bounds.append((img_name, 'Fovea', fovea_x, fovea_y, w, h))
        return out_of_bounds

    oob_train = check_bounds(train_annos, img_train_dir, 'train')
    oob_test = check_bounds(test_annos, img_test_dir, 'test')

    lines.append(f"  Training out-of-bounds coordinates: {len(oob_train)}")
    lines.append(f"  Testing out-of-bounds coordinates:  {len(oob_test)}")
    if oob_train:
        for err in oob_train:
            lines.append(f"    ERROR Train: {err}")
    if oob_test:
        for err in oob_test:
            lines.append(f"    ERROR Test: {err}")
    assert len(oob_train) == 0, f"Out of bounds in train: {oob_train}"
    assert len(oob_test) == 0, f"Out of bounds in test: {oob_test}"
    lines.append("  -> All 413 training coordinates and 103 testing coordinates are strictly within valid image bounds [0, W) x [0, H).")

    # 6. Summary Statistics of Ground Truth
    lines.append(f"\n[7] GROUND TRUTH SUMMARY STATISTICS (PIXELS):")
    lines.append("  TRAINING SET:")
    lines.append(f"    OD X:    Min={train_annos['od_x'].min():.1f}, Max={train_annos['od_x'].max():.1f}, Mean={train_annos['od_x'].mean():.1f}")
    lines.append(f"    OD Y:    Min={train_annos['od_y'].min():.1f}, Max={train_annos['od_y'].max():.1f}, Mean={train_annos['od_y'].mean():.1f}")
    lines.append(f"    Fovea X: Min={train_annos['fovea_x'].min():.1f}, Max={train_annos['fovea_x'].max():.1f}, Mean={train_annos['fovea_x'].mean():.1f}")
    lines.append(f"    Fovea Y: Min={train_annos['fovea_y'].min():.1f}, Max={train_annos['fovea_y'].max():.1f}, Mean={train_annos['fovea_y'].mean():.1f}")
    lines.append("  TESTING SET:")
    lines.append(f"    OD X:    Min={test_annos['od_x'].min():.1f}, Max={test_annos['od_x'].max():.1f}, Mean={test_annos['od_x'].mean():.1f}")
    lines.append(f"    OD Y:    Min={test_annos['od_y'].min():.1f}, Max={test_annos['od_y'].max():.1f}, Mean={test_annos['od_y'].mean():.1f}")
    lines.append(f"    Fovea X: Min={test_annos['fovea_x'].min():.1f}, Max={test_annos['fovea_x'].max():.1f}, Mean={test_annos['fovea_x'].mean():.1f}")
    lines.append(f"    Fovea Y: Min={test_annos['fovea_y'].min():.1f}, Max={test_annos['fovea_y'].max():.1f}, Mean={test_annos['fovea_y'].mean():.1f}")

    lines.append(f"\n[8] FINAL AUDIT VERDICT: PASSED (Integrity 100%)")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(report_text)

if __name__ == "__main__":
    verify_dataset()
