import os
import cv2
import shutil
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# NETRASETU - IDRiD DATA PREPARATION
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"

RAW_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "raw",
    "images",
    "A. Segmentation"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed"
)

IMAGE_SIZE = (512, 512)


# ============================================================
# RAW PATHS
# ============================================================

TRAIN_IMAGES = os.path.join(
    RAW_ROOT,
    "1. Original Images",
    "a. Training Set"
)

TEST_IMAGES = os.path.join(
    RAW_ROOT,
    "1. Original Images",
    "b. Testing Set"
)


TRAIN_MASK_ROOT = os.path.join(
    RAW_ROOT,
    "2. All Segmentation Groundtruths",
    "a. Training Set"
)

TEST_MASK_ROOT = os.path.join(
    RAW_ROOT,
    "2. All Segmentation Groundtruths",
    "b. Testing Set"
)


# ============================================================
# MASK PATHS
# ============================================================

TRAIN_MA = os.path.join(
    TRAIN_MASK_ROOT,
    "1. Microaneurysms"
)

TRAIN_HE = os.path.join(
    TRAIN_MASK_ROOT,
    "2. Haemorrhages"
)

TRAIN_EX = os.path.join(
    TRAIN_MASK_ROOT,
    "3. Hard Exudates"
)

TRAIN_SE = os.path.join(
    TRAIN_MASK_ROOT,
    "4. Soft Exudates"
)


TEST_MA = os.path.join(
    TEST_MASK_ROOT,
    "1. Microaneurysms"
)

TEST_HE = os.path.join(
    TEST_MASK_ROOT,
    "2. Haemorrhages"
)

TEST_EX = os.path.join(
    TEST_MASK_ROOT,
    "3. Hard Exudates"
)

TEST_SE = os.path.join(
    TEST_MASK_ROOT,
    "4. Soft Exudates"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            split,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            split,
            "masks"
        ),
        exist_ok=True
    )


# ============================================================
# READ IMAGE IDS
# ============================================================

def get_image_ids(folder):

    files = os.listdir(folder)

    ids = []

    for file in files:

        if file.lower().endswith(".jpg"):

            image_id = os.path.splitext(file)[0]

            ids.append(image_id)

    return sorted(ids)


train_ids = get_image_ids(
    TRAIN_IMAGES
)

test_ids = get_image_ids(
    TEST_IMAGES
)


print("\n============================================")
print("IDRiD DATASET")
print("============================================")

print(
    "Training images:",
    len(train_ids)
)

print(
    "Testing images:",
    len(test_ids)
)


# ============================================================
# SPLIT OFFICIAL TRAINING IMAGES
# 80% TRAIN
# 20% VALIDATION
# ============================================================

train_ids, val_ids = train_test_split(
    train_ids,
    test_size=0.20,
    random_state=42
)


print("\nSplit:")

print(
    "Train:",
    len(train_ids)
)

print(
    "Validation:",
    len(val_ids)
)

print(
    "Official Test:",
    len(test_ids)
)


# ============================================================
# FIND MASK
# ============================================================

def mask_path(folder, image_id, suffix):

    path = os.path.join(
        folder,
        image_id + suffix + ".tif"
    )

    if os.path.exists(path):

        return path

    return None


# ============================================================
# CREATE MULTI-CLASS MASK
#
# 0 = Background
# 1 = Microaneurysm
# 2 = Hemorrhage
# 3 = Hard Exudate
# 4 = Soft Exudate
# ============================================================

def create_mask(
    image_id,
    ma_dir,
    he_dir,
    ex_dir,
    se_dir
):

    # Find at least one mask
    reference_mask = None

    for folder, suffix in [
        (ma_dir, "_MA"),
        (he_dir, "_HE"),
        (ex_dir, "_EX"),
        (se_dir, "_SE")
    ]:

        path = mask_path(
            folder,
            image_id,
            suffix
        )

        if path is not None:

            reference_mask = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if reference_mask is not None:
                break


    if reference_mask is None:

        return None


    h, w = reference_mask.shape

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Microaneurysm = 1
    # --------------------------------------------------------

    ma = mask_path(
        ma_dir,
        image_id,
        "_MA"
    )

    if ma:

        img = cv2.imread(
            ma,
            cv2.IMREAD_GRAYSCALE
        )

        if img is not None:

            mask[img > 0] = 1


    # --------------------------------------------------------
    # Hemorrhage = 2
    # --------------------------------------------------------

    he = mask_path(
        he_dir,
        image_id,
        "_HE"
    )

    if he:

        img = cv2.imread(
            he,
            cv2.IMREAD_GRAYSCALE
        )

        if img is not None:

            mask[img > 0] = 2


    # --------------------------------------------------------
    # Hard Exudate = 3
    # --------------------------------------------------------

    ex = mask_path(
        ex_dir,
        image_id,
        "_EX"
    )

    if ex:

        img = cv2.imread(
            ex,
            cv2.IMREAD_GRAYSCALE
        )

        if img is not None:

            mask[img > 0] = 3


    # --------------------------------------------------------
    # Soft Exudate = 4
    # --------------------------------------------------------

    se = mask_path(
        se_dir,
        image_id,
        "_SE"
    )

    if se:

        img = cv2.imread(
            se,
            cv2.IMREAD_GRAYSCALE
        )

        if img is not None:

            mask[img > 0] = 4


    return mask


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_id,
    image_dir,
    ma_dir,
    he_dir,
    ex_dir,
    se_dir,
    split
):

    image_path = os.path.join(
        image_dir,
        image_id + ".jpg"
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "\nCould not read:",
            image_path
        )

        return False


    mask = create_mask(
        image_id,
        ma_dir,
        he_dir,
        ex_dir,
        se_dir
    )

    if mask is None:

        print(
            "\nNo mask found:",
            image_id
        )

        return False


    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Resize mask
    #
    # IMPORTANT:
    # Use nearest-neighbor so class labels
    # 0,1,2,3,4 are not mixed.
    # --------------------------------------------------------

    mask = cv2.resize(
        mask,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    output_image = os.path.join(
        OUTPUT_ROOT,
        split,
        "images",
        image_id + ".jpg"
    )


    cv2.imwrite(
        output_image,
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )


    # --------------------------------------------------------
    # Save mask
    # PNG is used because it preserves class values.
    # --------------------------------------------------------

    output_mask = os.path.join(
        OUTPUT_ROOT,
        split,
        "masks",
        image_id + ".png"
    )


    cv2.imwrite(
        output_mask,
        mask
    )


    return True


# ============================================================
# PROCESS TRAIN
# ============================================================

print("\nProcessing TRAIN...")

train_success = 0

for image_id in tqdm(train_ids):

    success = process_image(
        image_id,
        TRAIN_IMAGES,
        TRAIN_MA,
        TRAIN_HE,
        TRAIN_EX,
        TRAIN_SE,
        "train"
    )

    if success:
        train_success += 1


# ============================================================
# PROCESS VALIDATION
# ============================================================

print("\nProcessing VALIDATION...")

val_success = 0

for image_id in tqdm(val_ids):

    success = process_image(
        image_id,
        TRAIN_IMAGES,
        TRAIN_MA,
        TRAIN_HE,
        TRAIN_EX,
        TRAIN_SE,
        "val"
    )

    if success:
        val_success += 1


# ============================================================
# PROCESS OFFICIAL TEST
# ============================================================

print("\nProcessing OFFICIAL TEST...")

test_success = 0

for image_id in tqdm(test_ids):

    success = process_image(
        image_id,
        TEST_IMAGES,
        TEST_MA,
        TEST_HE,
        TEST_EX,
        TEST_SE,
        "test"
    )

    if success:
        test_success += 1


# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

with open(
    os.path.join(
        OUTPUT_ROOT,
        "dataset_split.txt"
    ),
    "w"
) as f:

    f.write(
        "IDRiD Dataset Split\n"
    )

    f.write(
        "===================\n\n"
    )

    f.write(
        f"Train: {train_success}\n"
    )

    f.write(
        f"Validation: {val_success}\n"
    )

    f.write(
        f"Official Test: {test_success}\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n============================================")
print("IDRiD PREPROCESSING COMPLETE")
print("============================================")

print(
    "Train images:",
    train_success
)

print(
    "Validation images:",
    val_success
)

print(
    "Test images:",
    test_success
)

print(
    "\nOutput:"
)

print(
    OUTPUT_ROOT
)

print(
    "\nMask classes:"
)

print(
    "0 = Background"
)

print(
    "1 = Microaneurysm"
)

print(
    "2 = Hemorrhage"
)

print(
    "3 = Hard Exudate"
)

print(
    "4 = Soft Exudate"
)

print(
    "============================================"
)