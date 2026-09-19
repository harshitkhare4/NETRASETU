import os
import cv2
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# NETRASETU - IDRiD U-NET V2
#
# 4 Independent Lesion Channels
#
# Channel 0 = Microaneurysm
# Channel 1 = Hemorrhage
# Channel 2 = Hard Exudate
# Channel 3 = Soft Exudate
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"

TRAIN_IMG_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed",
    "train",
    "images"
)

TRAIN_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed",
    "train",
    "masks"
)

VAL_IMG_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed",
    "val",
    "images"
)

VAL_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed",
    "val",
    "masks"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "idrid_v2"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n============================================")
print("NETRASETU IDRiD U-NET V2")
print("============================================")

print(
    "Device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory: %.2f GB"
        % (
            torch.cuda.get_device_properties(0)
            .total_memory / 1024**3
        )
    )


# ============================================================
# DATASET
# ============================================================

class IDRiDLesionDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted([
            file
            for file in os.listdir(image_dir)
            if file.lower().endswith(".jpg")
        ])


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_name = (
            os.path.splitext(
                image_name
            )[0]
            + ".png"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )


        if self.transform:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]


        # ----------------------------------------------------
        # Convert one multiclass mask into 4 independent masks
        #
        # Original:
        # 0 = background
        # 1 = MA
        # 2 = HE
        # 3 = Hard Exudate
        # 4 = Soft Exudate
        #
        # Output:
        # [MA, HE, EX, SE]
        # ----------------------------------------------------

        lesion_masks = torch.stack([
            (mask == 1).float(),
            (mask == 2).float(),
            (mask == 3).float(),
            (mask == 4).float()
        ])

        return image, lesion_masks


# ============================================================
# AUGMENTATION
# ============================================================

train_transform = A.Compose([

    A.HorizontalFlip(
        p=0.5
    ),

    A.VerticalFlip(
        p=0.5
    ),

    A.RandomRotate90(
        p=0.25
    ),

    A.Rotate(
        limit=15,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    ),

    A.Affine(
    translate_percent={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05)
    },
    scale=(
        0.90,
        1.10
    ),
    rotate=0,
    p=0.3
),

    A.Resize(
        512,
        512
    ),

    A.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),

    ToTensorV2()
])


val_transform = A.Compose([

    A.Resize(
        512,
        512
    ),

    A.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),

    ToTensorV2()
])


# ============================================================
# DATA
# ============================================================

train_dataset = IDRiDLesionDataset(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    train_transform
)

val_dataset = IDRiDLesionDataset(
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    val_transform
)

print(
    "\nTrain images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

BATCH_SIZE = 1

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


class UNetV2(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()


        # Encoder
        self.enc1 = DoubleConv(
            3,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )


        self.pool = nn.MaxPool2d(
            kernel_size=2
        )


        # Bottleneck
        self.bottleneck = DoubleConv(
            512,
            1024
        )


        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )


        # 4 independent lesion outputs
        self.final = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )


    def forward(self, x):

        # Encoder

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )


        # Bottleneck

        b = self.bottleneck(
            self.pool(e4)
        )


        # Decoder

        d4 = self.up4(
            b
        )

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(
            d4
        )


        d3 = self.up3(
            d4
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(
            d3
        )


        d2 = self.up2(
            d3
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(
            d2
        )


        d1 = self.up1(
            d2
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(
            d1
        )


        return self.final(
            d1
        )


# ============================================================
# MODEL
# ============================================================

model = UNetV2(
    num_classes=4
)

model = model.to(
    device
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

# Higher weights for the small lesion channels
pos_weights = torch.tensor(
    [
        8.0,   # Microaneurysm
        4.0,   # Hemorrhage
        2.0,   # Hard Exudate
        2.0    # Soft Exudate
    ],
    dtype=torch.float32,
    device=device
).view(1, 4, 1, 1)

bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=pos_weights
)


def dice_loss(
    predictions,
    targets,
    smooth=1.0
):

    predictions = torch.sigmoid(
        predictions
    )


    predictions = predictions.reshape(
        predictions.shape[0],
        predictions.shape[1],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        targets.shape[1],
        -1
    )


    intersection = (
        predictions * targets
    ).sum(
        dim=2
    )


    denominator = (
        predictions.sum(
            dim=2
        )
        +
        targets.sum(
            dim=2
        )
    )


    dice = (
        (2.0 * intersection + smooth)
        /
        (denominator + smooth)
    )


    return 1.0 - dice.mean()


def combined_loss(
    predictions,
    targets
):

    bce = bce_loss(
        predictions,
        targets
    )

    dice = dice_loss(
        predictions,
        targets
    )

    return (
        0.5 * bce
        +
        0.5 * dice
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4
)


# ============================================================
# DICE PER LESION
# ============================================================

def calculate_dice_per_class(
    predictions,
    targets
):

    predictions = (
        torch.sigmoid(
            predictions
        ) > 0.5
    ).float()


    results = []


    for class_id in range(4):

        pred = predictions[
            :,
            class_id,
            :,
            :
        ]

        target = targets[
            :,
            class_id,
            :,
            :
        ]


        intersection = (
            pred * target
        ).sum()
        

        denominator = (
            pred.sum()
            +
            target.sum()
        )


        dice = (
            (2 * intersection + 1e-7)
            /
            (denominator + 1e-7)
        )


        results.append(
            dice.item()
        )


    return results


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 50

best_val_dice = -1

best_model = copy.deepcopy(
    model.state_dict()
)


history = {
    "train_loss": [],
    "val_loss": [],
    "val_ma": [],
    "val_he": [],
    "val_hard_ex": [],
    "val_soft_ex": [],
    "val_mean_dice": []
}


for epoch in range(
    EPOCHS
):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "-" * 50
    )


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_train_loss = 0.0


    for images, masks in tqdm(
        train_loader,
        desc="Training"
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = combined_loss(
            outputs,
            masks
        )


        loss.backward()


        optimizer.step()


        running_train_loss += (
            loss.item()
            *
            images.size(0)
        )


    train_loss = (
        running_train_loss
        /
        len(train_dataset)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_val_loss = 0.0

    all_dice = []


    with torch.no_grad():

        for images, masks in tqdm(
            val_loader,
            desc="Validation"
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )


            outputs = model(
                images
            )


            loss = combined_loss(
                outputs,
                masks
            )


            running_val_loss += (
                loss.item()
                *
                images.size(0)
            )


            dice_values = calculate_dice_per_class(
                outputs,
                masks
            )


            all_dice.append(
                dice_values
            )


    val_loss = (
        running_val_loss
        /
        len(val_dataset)
    )


    all_dice = np.array(
        all_dice
    )


    mean_dice_per_class = (
        all_dice.mean(
            axis=0
        )
    )


    mean_dice = (
        mean_dice_per_class.mean()
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        mean_dice
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_ma"].append(
        mean_dice_per_class[0]
    )

    history["val_he"].append(
        mean_dice_per_class[1]
    )

    history["val_hard_ex"].append(
        mean_dice_per_class[2]
    )

    history["val_soft_ex"].append(
        mean_dice_per_class[3]
    )

    history["val_mean_dice"].append(
        mean_dice
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Microaneurysm Dice: "
        f"{mean_dice_per_class[0]:.4f}"
    )

    print(
        f"Hemorrhage Dice: "
        f"{mean_dice_per_class[1]:.4f}"
    )

    print(
        f"Hard Exudate Dice: "
        f"{mean_dice_per_class[2]:.4f}"
    )

    print(
        f"Soft Exudate Dice: "
        f"{mean_dice_per_class[3]:.4f}"
    )

    print(
        f"Mean Dice: "
        f"{mean_dice:.4f}"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if mean_dice > best_val_dice:

        best_val_dice = mean_dice

        best_model = copy.deepcopy(
            model.state_dict()
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    4,

                "classes": [
                    "microaneurysm",
                    "hemorrhage",
                    "hard_exudate",
                    "soft_exudate"
                ],

                "image_size":
                    512
            },

            os.path.join(
                MODEL_DIR,
                "NetraSetu_IDRiD_UNet_V2_best.pth"
            )
        )


        print(
            "✅ New best V2 model saved."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "num_classes":
            4,

        "classes": [
            "microaneurysm",
            "hemorrhage",
            "hard_exudate",
            "soft_exudate"
        ],

        "image_size":
            512
    },

    os.path.join(
        MODEL_DIR,
        "NetraSetu_IDRiD_UNet_V2_final.pth"
    )
)


# ============================================================
# TRAINING CURVES
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1
)


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "IDRiD U-Net V2 Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "loss_curve.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Mean Dice
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history["val_mean_dice"],
    label="Mean Validation Dice"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Dice"
)

plt.title(
    "IDRiD U-Net V2 Mean Dice"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "mean_dice_curve.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Lesion Dice
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs_range,
    history["val_ma"],
    label="Microaneurysm"
)

plt.plot(
    epochs_range,
    history["val_he"],
    label="Hemorrhage"
)

plt.plot(
    epochs_range,
    history["val_hard_ex"],
    label="Hard Exudate"
)

plt.plot(
    epochs_range,
    history["val_soft_ex"],
    label="Soft Exudate"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Dice"
)

plt.title(
    "IDRiD U-Net V2 Lesion Dice"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "lesion_dice_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINISH
# ============================================================

print("\n============================================")
print("IDRiD U-NET V2 TRAINING COMPLETE")
print("============================================")

print(
    "Best Validation Mean Dice:",
    f"{best_val_dice:.4f}"
)

print(
    "\nBest Model:"
)

print(
    os.path.join(
        MODEL_DIR,
        "NetraSetu_IDRiD_UNet_V2_best.pth"
    )
)

print(
    "\nFinal Model:"
)

print(
    os.path.join(
        MODEL_DIR,
        "NetraSetu_IDRiD_UNet_V2_final.pth"
    )
)

print(
    "\nResults:"
)

print(
    RESULT_DIR
)