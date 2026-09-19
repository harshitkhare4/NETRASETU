import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# NETRASETU - IDRiD U-NET V2 THRESHOLD OPTIMIZATION
# Validation only -> optimize 4 lesion thresholds independently
# ============================================================

ROOT = r"C:\NetraSetu"
VAL_IMG_DIR = os.path.join(ROOT, "data", "IDRiD", "processed", "val", "images")
VAL_MASK_DIR = os.path.join(ROOT, "data", "IDRiD", "processed", "val", "masks")

MODEL_PATH = os.path.join(
    ROOT, "models", "NetraSetu_IDRiD_UNet_V2_best.pth"
)

OUTPUT_DIR = os.path.join(ROOT, "results", "idrid_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD_FILE = os.path.join(
    OUTPUT_DIR, "optimized_thresholds.txt"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LESIONS = [
    "Microaneurysm",
    "Hemorrhage",
    "Hard Exudate",
    "Soft Exudate"
]


# ============================================================
# U-NET V2
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# DATASET
# ============================================================

class IDRiDValidationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(
                f"No validation images found in:\n{image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        filename = self.images[idx]

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(filename)[0] + ".png"
        )

        image = cv2.imread(img_path)

        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )
        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (image - mean) / std

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(
                f"Cannot read mask: {mask_path}"
            )

        # Original multiclass mask:
        # 0 background
        # 1 MA
        # 2 HE
        # 3 Hard EX
        # 4 Soft EX

        mask = np.stack([
            (mask == 1),
            (mask == 2),
            (mask == 3),
            (mask == 4)
        ], axis=0).astype(np.float32)

        mask = torch.from_numpy(mask)

        return image, mask, filename


# ============================================================
# METRICS
# ============================================================

def dice_score(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()

    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return (2.0 * intersection) / denominator


def iou_score(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()

    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("============================================")
print("NETRASETU IDRiD V2 THRESHOLD OPTIMIZATION")
print("============================================")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()
print("Validation images loading...")

dataset = IDRiDValidationDataset(
    VAL_IMG_DIR,
    VAL_MASK_DIR
)

print(f"Validation images: {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print()
print("Loading V2 model...")

model = UNetV2(
    in_channels=3,
    out_channels=4
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

# ============================================================
# COLLECT VALIDATION PROBABILITIES
# ============================================================

all_probs = []
all_targets = []

print()
print("Running validation inference...")

with torch.no_grad():

    for images, masks, _ in tqdm(loader):

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        all_probs.append(
            probs.cpu().numpy()[0]
        )

        all_targets.append(
            masks.numpy()[0]
        )

all_probs = np.stack(all_probs, axis=0)
all_targets = np.stack(all_targets, axis=0)

print()
print("Validation inference complete.")

# ============================================================
# THRESHOLD SEARCH
# ============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.02
)

best_thresholds = {}

print()
print("============================================")
print("OPTIMIZING THRESHOLDS")
print("============================================")

for lesion_idx, lesion_name in enumerate(LESIONS):

    best_threshold = 0.50
    best_dice = -1.0
    best_iou = 0.0

    print()
    print(f"{lesion_name}")

    for threshold in thresholds:

        pred = (
            all_probs[:, lesion_idx] >= threshold
        )

        target = (
            all_targets[:, lesion_idx] > 0.5
        )

        dice_values = []
        iou_values = []

        for i in range(len(pred)):

            d = dice_score(
                pred[i],
                target[i]
            )

            j = iou_score(
                pred[i],
                target[i]
            )

            dice_values.append(d)
            iou_values.append(j)

        mean_dice = np.mean(dice_values)
        mean_iou = np.mean(iou_values)

        if mean_dice > best_dice:

            best_dice = mean_dice
            best_iou = mean_iou
            best_threshold = float(threshold)

    best_thresholds[lesion_name] = best_threshold

    print(
        f"Best Threshold : {best_threshold:.2f}"
    )
    print(
        f"Validation Dice: {best_dice:.4f}"
    )
    print(
        f"Validation IoU : {best_iou:.4f}"
    )

# ============================================================
# SAVE
# ============================================================

print()
print("============================================")
print("FINAL OPTIMIZED THRESHOLDS")
print("============================================")

with open(
    THRESHOLD_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NETRASETU IDRiD U-NET V2\n"
        "Validation-based optimized thresholds\n"
        "Threshold criterion: maximum mean Dice\n\n"
    )

    for lesion_name in LESIONS:

        threshold = best_thresholds[lesion_name]

        print(
            f"{lesion_name:<18}: {threshold:.2f}"
        )

        f.write(
            f"{lesion_name}: {threshold:.2f}\n"
        )

print()
print("Thresholds saved:")
print(THRESHOLD_FILE)

print()
print("============================================")
print("DONE")
print("============================================")