import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm


# ============================================================
# NETRASETU - IDRiD MULTI-CLASS U-NET
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"

TRAIN_IMG = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed",
    "train",
    "images"
)

TRAIN_MASK = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed",
    "train",
    "masks"
)

VAL_IMG = os.path.join(
    PROJECT_ROOT,
    "data",
    "IDRiD",
    "processed",
    "val",
    "images"
)

VAL_MASK = os.path.join(
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
    "idrid"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n============================================")
print("NETRASETU IDRiD U-NET TRAINING")
print("============================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATASET
# ============================================================

class IDRiDDataset(Dataset):

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
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg")
        ])


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_name = os.path.splitext(
            image_name
        )[0] + ".png"

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        image = cv2.imread(
            image_path
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if self.transform:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]


        return image, mask.long()


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

    A.Rotate(
        limit=15,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    ),

    A.Resize(
        512,
        512
    ),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()
])


val_transform = A.Compose([

    A.Resize(
        512,
        512
    ),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = IDRiDDataset(
    TRAIN_IMG,
    TRAIN_MASK,
    train_transform
)

val_dataset = IDRiDDataset(
    VAL_IMG,
    VAL_MASK,
    val_transform
)


print("\nTrain images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
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
                padding=1
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
                padding=1
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


class UNet(nn.Module):

    def __init__(
        self,
        num_classes=5
    ):

        super().__init__()


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
            2
        )


        self.bottleneck = DoubleConv(
            512,
            1024
        )


        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )


        self.final = nn.Conv2d(
            64,
            num_classes,
            1
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
# MODEL
# ============================================================

model = UNet(
    num_classes=5
)

model = model.to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = torch.tensor(
    [
        0.1,
        2.0,
        2.0,
        1.5,
        1.5
    ],
    dtype=torch.float32
).to(device)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
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
# DICE FUNCTION
# ============================================================

def dice_score(
    predictions,
    targets,
    num_classes=5
):

    predictions = torch.argmax(
        predictions,
        dim=1
    )

    dice_scores = []


    for class_id in range(
        1,
        num_classes
    ):

        pred = (
            predictions == class_id
        ).float()

        true = (
            targets == class_id
        ).float()


        intersection = (
            pred * true
        ).sum()


        denominator = (
            pred.sum()
            +
            true.sum()
        )


        if denominator == 0:

            continue


        dice = (
            2 * intersection
            /
            (denominator + 1e-7)
        )


        dice_scores.append(
            dice.item()
        )


    if len(dice_scores) == 0:

        return 0.0


    return np.mean(
        dice_scores
    )


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 30

best_val_loss = float(
    "inf"
)


for epoch in range(
    EPOCHS
):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        "-" * 50
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0


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


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            masks
        )


        loss.backward()


        optimizer.step()


        train_loss += (
            loss.item()
            *
            images.size(0)
        )


    train_loss /= len(
        train_dataset
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_dice = []


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


            loss = criterion(
                outputs,
                masks
            )


            val_loss += (
                loss.item()
                *
                images.size(0)
            )


            dice = dice_score(
                outputs,
                masks
            )


            val_dice.append(
                dice
            )


    val_loss /= len(
        val_dataset
    )


    mean_dice = np.mean(
        val_dice
    )


    print(
        f"\nTrain Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Dice: "
        f"{mean_dice:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "num_classes":
                    5,

                "classes": [
                    "background",
                    "microaneurysm",
                    "hemorrhage",
                    "hard_exudate",
                    "soft_exudate"
                ]
            },

            os.path.join(
                MODEL_DIR,
                "NetraSetu_IDRiD_UNet_best.pth"
            )
        )


        print(
            "✅ Best U-Net saved."
        )


# ============================================================
# FINAL
# ============================================================

print(
    "\n============================================"
)

print(
    "IDRiD U-NET TRAINING COMPLETE"
)

print(
    "============================================"
)

print(
    "Model:"
)

print(
    os.path.join(
        MODEL_DIR,
        "NetraSetu_IDRiD_UNet_best.pth"
    )
)