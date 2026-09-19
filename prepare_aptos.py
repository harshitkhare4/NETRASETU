import os
import cv2
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# NETRASETU - APTOS PREPROCESSING + QUALITY FILTER + SPLIT
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"

CSV_PATH = os.path.join(
    PROJECT_ROOT, "data", "train.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT, "data", "train_images"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, "data", "APTOS", "processed"
)

REJECTED_DIR = os.path.join(
    OUTPUT_ROOT, "rejected"
)

IMAGE_SIZE = (512, 512)

# Conservative quality thresholds
# These are intentionally not aggressive.
MIN_RETINAL_AREA = 0.30
MIN_CONTRAST = 15.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 235.0

# Blur threshold
# We use a dataset-relative threshold later as well.
MIN_LAPLACIAN_VARIANCE = 8.0


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    for label in range(5):
        os.makedirs(
            os.path.join(OUTPUT_ROOT, split, str(label)),
            exist_ok=True
        )

os.makedirs(REJECTED_DIR, exist_ok=True)


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading APTOS CSV...")

df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
print(df.columns.tolist())

required_columns = {"id_code", "diagnosis"}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print(f"\nTotal rows in CSV: {len(df)}")


# ============================================================
# IMAGE PATH SEARCH
# ============================================================

def find_image(image_id):

    extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG"
    ]

    for ext in extensions:

        path = os.path.join(
            IMAGE_DIR,
            str(image_id) + ext
        )

        if os.path.exists(path):
            return path

    return None


df["image_path"] = df["id_code"].apply(find_image)

missing_images = df["image_path"].isna().sum()

print(f"Images not found: {missing_images}")

df = df[df["image_path"].notna()].copy()

print(f"Images available: {len(df)}")


# ============================================================
# FUNDUS CROPPING
# ============================================================

def crop_black_border(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Find non-black retinal region
    mask = gray > 10

    coords = np.column_stack(
        np.where(mask)
    )

    if len(coords) == 0:
        return image, 0.0

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    original_area = image.shape[0] * image.shape[1]

    cropped_area = cropped.shape[0] * cropped.shape[1]

    retinal_area_ratio = (
        cropped_area / original_area
    )

    return cropped, retinal_area_ratio


# ============================================================
# QUALITY ANALYSIS
# ============================================================

def calculate_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Focus / blur
    laplacian_variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    return (
        laplacian_variance,
        brightness,
        contrast
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # 1. Crop black background
    # --------------------------------------------------------

    image, retinal_area = crop_black_border(image)

    # --------------------------------------------------------
    # 2. Resize
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 3. Mild denoising
    # --------------------------------------------------------

    # Bilateral filtering removes small noise while
    # preserving lesion boundaries.
    image = cv2.bilateralFilter(
        image,
        d=5,
        sigmaColor=30,
        sigmaSpace=30
    )

    # --------------------------------------------------------
    # 4. CLAHE
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        [l_channel, a_channel, b_channel]
    )

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------------
    # 5. Mild illumination normalization
    # --------------------------------------------------------

    image_float = image.astype(
        np.float32
    )

    background = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=25
    )

    normalized = (
        image_float / (background + 1.0)
    )

    normalized = (
        normalized * 128.0
    )

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    image = normalized

    return image, retinal_area


# ============================================================
# FIRST PASS - QUALITY MEASUREMENTS
# ============================================================

print("\nCalculating image quality...")

quality_records = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    image_path = row["image_path"]

    image = cv2.imread(
        image_path
    )

    if image is None:

        quality_records.append({
            "id_code": row["id_code"],
            "diagnosis": row["diagnosis"],
            "status": "rejected",
            "reason": "unreadable",
            "sharpness": 0,
            "brightness": 0,
            "contrast": 0,
            "retinal_area": 0
        })

        continue

    cropped, retinal_area = crop_black_border(
        image
    )

    sharpness, brightness, contrast = (
        calculate_quality(cropped)
    )

    quality_records.append({
        "id_code": row["id_code"],
        "diagnosis": int(row["diagnosis"]),
        "status": "candidate",
        "reason": "",
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "retinal_area": retinal_area
    })


quality_df = pd.DataFrame(
    quality_records
)


# ============================================================
# DATASET-RELATIVE BLUR THRESHOLD
# ============================================================

valid_sharpness = quality_df.loc[
    quality_df["status"] == "candidate",
    "sharpness"
]

if len(valid_sharpness) > 0:

    # Conservative bottom 3%
    relative_blur_threshold = np.percentile(
        valid_sharpness,
        3
    )

else:

    relative_blur_threshold = (
        MIN_LAPLACIAN_VARIANCE
    )

blur_threshold = max(
    MIN_LAPLACIAN_VARIANCE,
    relative_blur_threshold
)

print(
    f"\nBlur threshold: {blur_threshold:.2f}"
)


# ============================================================
# QUALITY FILTER
# ============================================================

def quality_rejection_reason(row):

    if row["status"] == "rejected":
        return row["reason"]

    reasons = []

    if row["sharpness"] < blur_threshold:
        reasons.append("severe_blur")

    if row["brightness"] < MIN_BRIGHTNESS:
        reasons.append("too_dark")

    if row["brightness"] > MAX_BRIGHTNESS:
        reasons.append("too_bright")

    if row["contrast"] < MIN_CONTRAST:
        reasons.append("low_contrast")

    if row["retinal_area"] < MIN_RETINAL_AREA:
        reasons.append("insufficient_retinal_area")

    if len(reasons) > 0:
        return "|".join(reasons)

    return ""


quality_df["reason"] = quality_df.apply(
    quality_rejection_reason,
    axis=1
)

quality_df["status"] = np.where(
    quality_df["reason"] == "",
    "accepted",
    "rejected"
)


# ============================================================
# PRINT QUALITY REPORT
# ============================================================

print("\n================ QUALITY REPORT ================")

print(
    quality_df["status"].value_counts()
)

print("\nRejected by reason:")

print(
    quality_df.loc[
        quality_df["status"] == "rejected",
        "reason"
    ].value_counts()
)

quality_df.to_csv(
    os.path.join(
        OUTPUT_ROOT,
        "quality_report.csv"
    ),
    index=False
)


# ============================================================
# SAVE REJECTED IMAGES
# ============================================================

print(
    "\nSaving rejected images..."
)

for _, row in tqdm(
    quality_df[
        quality_df["status"] == "rejected"
    ].iterrows()
):

    image_path = find_image(
        row["id_code"]
    )

    if image_path is None:
        continue

    extension = os.path.splitext(
        image_path
    )[1]

    destination = os.path.join(
        REJECTED_DIR,
        f'{row["id_code"]}_{row["reason"]}{extension}'
    )

    shutil.copy2(
        image_path,
        destination
    )


# ============================================================
# KEEP ONLY ACCEPTED DATA
# ============================================================

accepted_ids = set(
    quality_df.loc[
        quality_df["status"] == "accepted",
        "id_code"
    ]
)

df = df[
    df["id_code"].isin(
        accepted_ids
    )
].copy()

print(
    f"\nAccepted images: {len(df)}"
)

print(
    f"Rejected images: {len(quality_df) - len(df)}"
)


# ============================================================
# STRATIFIED SPLIT 70 / 15 / 15
# ============================================================

print(
    "\nCreating stratified dataset split..."
)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["diagnosis"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["diagnosis"],
    random_state=42
)

print("\nSplit sizes:")

print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# SAVE PROCESSED IMAGES
# ============================================================

def process_and_save(dataframe, split_name):

    print(
        f"\nProcessing {split_name} images..."
    )

    for _, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe)
    ):

        image_path = row["image_path"]

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        try:

            processed, _ = preprocess_image(
                image
            )

            label = int(
                row["diagnosis"]
            )

            output_folder = os.path.join(
                OUTPUT_ROOT,
                split_name,
                str(label)
            )

            output_path = os.path.join(
                output_folder,
                f'{row["id_code"]}.jpg'
            )

            cv2.imwrite(
                output_path,
                processed,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

        except Exception as e:

            print(
                f"\nError processing {row['id_code']}: {e}"
            )


process_and_save(
    train_df,
    "train"
)

process_and_save(
    val_df,
    "val"
)

process_and_save(
    test_df,
    "test"
)


# ============================================================
# SAVE SPLIT CSV FILES
# ============================================================

train_df.to_csv(
    os.path.join(
        OUTPUT_ROOT,
        "train_split.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        OUTPUT_ROOT,
        "val_split.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_ROOT,
        "test_split.csv"
    ),
    index=False
)


# ============================================================
# FINAL CLASS DISTRIBUTION
# ============================================================

print(
    "\n================ FINAL DATASET ================"
)

for split_name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(
        f"\n{split_name}"
    )

    print(
        split_df["diagnosis"]
        .value_counts()
        .sort_index()
    )


print(
    "\n==============================================="
)

print(
    "NETRASETU APTOS PREPROCESSING COMPLETE"
)

print(
    f"Processed dataset:\n{OUTPUT_ROOT}"
)

print(
    f"Quality report:\n"
    f"{os.path.join(OUTPUT_ROOT, 'quality_report.csv')}"
)

print(
    f"Rejected images:\n{REJECTED_DIR}"
)

print(
    "==============================================="
)