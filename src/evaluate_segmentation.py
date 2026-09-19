import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# NETRASETU - IDRiD U-NET V2 OFFICIAL TEST
# Uses validation-optimized, LOCKED lesion thresholds
# ============================================================

ROOT = r"C:\NetraSetu"

TEST_IMG_DIR = os.path.join(
    ROOT, "data", "IDRiD", "processed", "test", "images"
)

TEST_MASK_DIR = os.path.join(
    ROOT, "data", "IDRiD", "processed", "test", "masks"
)

MODEL_PATH = os.path.join(
    ROOT, "models", "NetraSetu_IDRiD_UNet_V2_best.pth"
)

THRESHOLD_FILE = os.path.join(
    ROOT, "results", "idrid_v2", "optimized_thresholds.txt"
)

OUTPUT_DIR = os.path.join(
    ROOT, "results", "idrid_v2", "test_outputs_optimized"
)

RESULT_FILE = os.path.join(
    ROOT, "results", "idrid_v2", "v2_test_results_optimized.txt"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LESIONS = [
    "Microaneurysm",
    "Hemorrhage",
    "Hard Exudate",
    "Soft Exudate"
]


# ============================================================
# LOCKED VALIDATION THRESHOLDS
# ============================================================

THRESHOLDS = {
    "Microaneurysm": 0.34,
    "Hemorrhage": 0.30,
    "Hard Exudate": 0.64,
    "Soft Exudate": 0.64
}


# ============================================================
# U-NET V2
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
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

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec1 = DoubleConv(128, 64)

        # IMPORTANT:
        # checkpoint uses "final.weight" and "final.bias"
        self.final = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat(
            [d4, e4],
            dim=1
        )
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat(
            [d3, e3],
            dim=1
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )
        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# DATASET
# ============================================================

class IDRiDTestDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        if len(self.images) == 0:
            raise RuntimeError(
                f"No test images found in:\n{image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        filename = self.images[idx]

        img_path = os.path.join(
            self.image_dir,
            filename
        )

        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(filename)[0] + ".png"
        )

        image = cv2.imread(img_path)

        if image is None:
            raise RuntimeError(
                f"Cannot read image:\n{img_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        ) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (
            (image - mean) / std
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Cannot read mask:\n{mask_path}"
            )

        # 0 = background
        # 1 = Microaneurysm
        # 2 = Hemorrhage
        # 3 = Hard Exudate
        # 4 = Soft Exudate

        mask = np.stack([
            mask == 1,
            mask == 2,
            mask == 3,
            mask == 4
        ], axis=0).astype(
            np.float32
        )

        mask = torch.from_numpy(mask)

        return image, mask, filename


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(
        pred,
        target
    ).sum()

    fp = np.logical_and(
        pred,
        np.logical_not(target)
    ).sum()

    fn = np.logical_and(
        np.logical_not(pred),
        target
    ).sum()

    dice_den = (
        2 * tp + fp + fn
    )

    iou_den = (
        tp + fp + fn
    )

    precision_den = (
        tp + fp
    )

    recall_den = (
        tp + fn
    )

    dice = (
        (2 * tp) / dice_den
        if dice_den > 0 else 1.0
    )

    iou = (
        tp / iou_den
        if iou_den > 0 else 1.0
    )

    precision = (
        tp / precision_den
        if precision_den > 0 else 0.0
    )

    recall = (
        tp / recall_den
        if recall_den > 0 else 0.0
    )

    return (
        dice,
        iou,
        precision,
        recall
    )


# ============================================================
# VISUAL OVERLAY
# ============================================================

def create_overlay(
    image_rgb,
    prediction_masks
):

    image = image_rgb.copy()

    overlay = image.copy()

    # Red = Microaneurysm
    overlay[
        prediction_masks[0]
    ] = [255, 0, 0]

    # Green = Hemorrhage
    overlay[
        prediction_masks[1]
    ] = [0, 255, 0]

    # Blue = Hard Exudate
    overlay[
        prediction_masks[2]
    ] = [0, 0, 255]

    # Yellow = Soft Exudate
    overlay[
        prediction_masks[3]
    ] = [255, 255, 0]

    result = cv2.addWeighted(
        image,
        0.65,
        overlay,
        0.35,
        0
    )

    return result


# ============================================================
# MAIN
# ============================================================

print()
print("============================================")
print("NETRASETU IDRiD U-NET V2 OPTIMIZED TEST")
print("============================================")

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()
print("Validation-optimized thresholds:")

for lesion in LESIONS:
    print(
        f"{lesion:<18}: "
        f"{THRESHOLDS[lesion]:.2f}"
    )

print()
print(
    "Official test images loading..."
)

dataset = IDRiDTestDataset(
    TEST_IMG_DIR,
    TEST_MASK_DIR
)

print(
    f"Official test images: {len(dataset)}"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ============================================================
# LOAD MODEL
# ============================================================

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
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)
model.eval()

print(
    "Model loaded successfully."
)

# ============================================================
# TEST
# ============================================================

metric_values = {
    lesion: {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": []
    }
    for lesion in LESIONS
}

print()
print(
    "Running official IDRiD test with "
    "locked validation thresholds..."
)

with torch.no_grad():

    for images, masks, filenames in tqdm(
        loader
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
        ).cpu().numpy()[0]

        targets = masks.numpy()[0]

        prediction_masks = []

        for lesion_idx, lesion_name in enumerate(
            LESIONS
        ):

            threshold = THRESHOLDS[
                lesion_name
            ]

            prediction = (
                probabilities[
                    lesion_idx
                ] >= threshold
            )

            target = (
                targets[
                    lesion_idx
                ] > 0.5
            )

            dice, iou, precision, recall = (
                calculate_metrics(
                    prediction,
                    target
                )
            )

            metric_values[
                lesion_name
            ]["dice"].append(dice)

            metric_values[
                lesion_name
            ]["iou"].append(iou)

            metric_values[
                lesion_name
            ]["precision"].append(precision)

            metric_values[
                lesion_name
            ]["recall"].append(recall)

            prediction_masks.append(
                prediction
            )

        # Save visual overlay
        img_path = os.path.join(
            TEST_IMG_DIR,
            filenames[0]
        )

        original = cv2.imread(
            img_path
        )

        original_rgb = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        overlay = create_overlay(
            original_rgb,
            prediction_masks
        )

        output_name = (
            os.path.splitext(
                filenames[0]
            )[0]
            + "_optimized_overlay.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )
        )

# ============================================================
# RESULTS
# ============================================================

print()
print("============================================")
print("FINAL IDRiD V2 OPTIMIZED TEST RESULTS")
print("============================================")

mean_dice_values = []
mean_iou_values = []

result_lines = []

result_lines.append(
    "NETRASETU IDRiD U-NET V2 OPTIMIZED TEST"
)

result_lines.append(
    "Thresholds selected using validation set only"
)

result_lines.append("")

for lesion in LESIONS:

    dice = np.mean(
        metric_values[
            lesion
        ]["dice"]
    )

    iou = np.mean(
        metric_values[
            lesion
        ]["iou"]
    )

    precision = np.mean(
        metric_values[
            lesion
        ]["precision"]
    )

    recall = np.mean(
        metric_values[
            lesion
        ]["recall"]
    )

    mean_dice_values.append(
        dice
    )

    mean_iou_values.append(
        iou
    )

    print()
    print(lesion)
    print(
        f"Threshold : "
        f"{THRESHOLDS[lesion]:.2f}"
    )
    print(
        f"Dice      : {dice:.4f}"
    )
    print(
        f"IoU       : {iou:.4f}"
    )
    print(
        f"Precision : {precision:.4f}"
    )
    print(
        f"Recall    : {recall:.4f}"
    )

    result_lines.extend([
        lesion,
        f"Threshold : {THRESHOLDS[lesion]:.2f}",
        f"Dice      : {dice:.4f}",
        f"IoU       : {iou:.4f}",
        f"Precision : {precision:.4f}",
        f"Recall    : {recall:.4f}",
        ""
    ])

mean_dice = np.mean(
    mean_dice_values
)

mean_iou = np.mean(
    mean_iou_values
)

print()
print("============================================")
print(
    f"Mean Dice: {mean_dice:.4f}"
)
print(
    f"Mean IoU : {mean_iou:.4f}"
)
print("============================================")

result_lines.extend([
    "============================================",
    f"Mean Dice: {mean_dice:.4f}",
    f"Mean IoU : {mean_iou:.4f}",
    "============================================",
    "",
    "Locked thresholds:",
    "Microaneurysm: 0.34",
    "Hemorrhage: 0.30",
    "Hard Exudate: 0.64",
    "Soft Exudate: 0.64"
])

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(result_lines)
    )

print()
print("Results saved:")
print(RESULT_FILE)

print()
print("Visual predictions:")
print(OUTPUT_DIR)

print()
print("============================================")
print("DONE")
print("============================================")