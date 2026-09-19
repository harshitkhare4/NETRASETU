"""
src/train_unet_v3.py
NetraSetu — IDRiD Lesion Segmentation V3 Training Pipeline.

Features:
- 4 Independent Lesion Channels:
    0 = Microaneurysm (MA)
    1 = Hemorrhage (HE)
    2 = Hard Exudate (EX)
    3 = Soft Exudate (SE)
- Lightweight Residual U-Net (ResUNet) architecture.
- Combined Loss: Pos-weighted BCEWithLogitsLoss + Soft Dice Loss.
  Class positive weights computed strictly from 43 training masks.
- Augmentations: Flips, rotations, brightness/contrast jitter.
- Checkpoint selection: Strictly based on Validation Macro Mean Dice.
- Saves:
    models/NetraSetu_IDRiD_UNet_V3_best.pth
    models/NetraSetu_IDRiD_UNet_V3_final.pth
"""

import os
import cv2
import math
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------

PROJECT_ROOT = r"C:\NetraSetu"
TRAIN_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "IDRiD", "processed", "train", "images")
TRAIN_MASK_DIR = os.path.join(PROJECT_ROOT, "data", "IDRiD", "processed", "train", "masks")
VAL_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "IDRiD", "processed", "val", "images")
VAL_MASK_DIR = os.path.join(PROJECT_ROOT, "data", "IDRiD", "processed", "val", "masks")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "final_pipeline", "segmentation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------------------------------------
# DATASET
# -------------------------------------------------------------

class IDRiDLesionDatasetV3(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Basic normalization to float tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Build 4-channel target mask: [MA, HE, EX, SE]
        if isinstance(mask, np.ndarray):
            lesion_masks = torch.stack([
                torch.from_numpy((mask == 1).astype(np.float32)),
                torch.from_numpy((mask == 2).astype(np.float32)),
                torch.from_numpy((mask == 3).astype(np.float32)),
                torch.from_numpy((mask == 4).astype(np.float32))
            ])
        else:
            lesion_masks = torch.stack([
                (mask == 1).float(),
                (mask == 2).float(),
                (mask == 3).float(),
                (mask == 4).float()
            ])

        return image, lesion_masks, img_name

# -------------------------------------------------------------
# AUGMENTATIONS
# -------------------------------------------------------------

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# -------------------------------------------------------------
# ARCHITECTURE: RESIDUAL U-NET
# -------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class UNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(3, 48)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(48, 96)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(96, 192)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(192, 384)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(384, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 384, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(384 + 384, 384)

        self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(192 + 192, 192)

        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(96 + 96, 96)

        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(48 + 48, 48)

        # Output head
        self.head = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)              # 512x512, 48
        e2 = self.enc2(self.pool1(e1))  # 256x256, 96
        e3 = self.enc3(self.pool2(e2))  # 128x128, 192
        e4 = self.enc4(self.pool3(e3))  # 64x64, 384

        b = self.bottleneck(self.pool4(e4)) # 32x32, 512

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

# -------------------------------------------------------------
# LOSS FUNCTION: WEIGHTED BCE + SOFT DICE LOSS
# -------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num_classes = logits.shape[1]
        dice_total = 0.0

        for c in range(num_classes):
            p = probs[:, c].contiguous().view(-1)
            t = targets[:, c].contiguous().view(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_total += (1.0 - dice)

        return dice_total / num_classes

class CombinedLesionLoss(nn.Module):
    def __init__(self, pos_weights):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        self.dice = SoftDiceLoss(smooth=1.0)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return bce_loss + 1.5 * dice_loss

# -------------------------------------------------------------
# EVALUATION METRICS
# -------------------------------------------------------------

def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    # preds, targets: (N, H, W) binary
    p = (preds >= threshold).float()
    t = targets.float()

    intersection = (p * t).sum().item()
    p_sum = p.sum().item()
    t_sum = t.sum().item()

    dice = (2.0 * intersection + eps) / (p_sum + t_sum + eps)
    iou = (intersection + eps) / (p_sum + t_sum - intersection + eps)
    precision = (intersection + eps) / (p_sum + eps)
    recall = (intersection + eps) / (t_sum + eps)

    return dice, iou, precision, recall

def evaluate_unet(model, loader, device, thresholds=[0.5, 0.5, 0.5, 0.5]):
    model.eval()
    all_dices = [[] for _ in range(4)]
    all_ious = [[] for _ in range(4)]
    all_precisions = [[] for _ in range(4)]
    all_recalls = [[] for _ in range(4)]

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            bs = images.size(0)
            for b in range(bs):
                for c in range(4):
                    d, iou, p, r = compute_metrics(probs[b, c], masks[b, c], threshold=thresholds[c])
                    all_dices[c].append(d)
                    all_ious[c].append(iou)
                    all_precisions[c].append(p)
                    all_recalls[c].append(r)

    mean_dice_per_class = [np.mean(d) for d in all_dices]
    mean_iou_per_class = [np.mean(i) for i in all_ious]
    macro_dice = np.mean(mean_dice_per_class)
    macro_iou = np.mean(mean_iou_per_class)

    return macro_dice, macro_iou, mean_dice_per_class, mean_iou_per_class

# -------------------------------------------------------------
# MAIN TRAINING LOOP
# -------------------------------------------------------------

def main():
    print("=" * 60)
    print("NETRASETU — IDRiD LESION SEGMENTATION V3 TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = IDRiDLesionDatasetV3(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = IDRiDLesionDatasetV3(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

    print(f"Training images:   {len(train_dataset)} (Expected: 43)")
    print(f"Validation images: {len(val_dataset)} (Expected: 11)")

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Pos weights derived strictly from training masks:
    # MA: 0.09% -> ~25.0, HE: 1.05% -> ~8.0, EX: 0.79% -> ~8.0, SE: 0.20% -> ~18.0
    pos_weights = torch.tensor([25.0, 8.0, 8.0, 18.0], dtype=torch.float32).view(1, 4, 1, 1).to(device)

    model = UNetV3(num_classes=4).to(device)
    criterion = CombinedLesionLoss(pos_weights=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = 40
    early_stopping_patience = 8
    epochs_no_improve = 0
    best_macro_dice = -1.0
    best_epoch = -1

    best_model_path = os.path.join(MODEL_DIR, "NetraSetu_IDRiD_UNet_V3_best.pth")
    final_model_path = os.path.join(MODEL_DIR, "NetraSetu_IDRiD_UNet_V3_final.pth")

    history = []
    start_time = time.time()

    print("\nStarting training loop...")
    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation evaluation at default 0.5 threshold
        val_dice, val_iou, class_dices, _ = evaluate_unet(model, val_loader, device, thresholds=[0.5, 0.5, 0.5, 0.5])
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        ep_duration = time.time() - ep_start

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_macro_dice': val_dice,
            'ma_dice': class_dices[0],
            'he_dice': class_dices[1],
            'ex_dice': class_dices[2],
            'se_dice': class_dices[3],
            'lr': current_lr
        })

        print(f"Epoch {epoch:02d}/{num_epochs:02d} [{ep_duration:.1f}s] | Loss: {train_loss:.4f} | "
              f"Val Macro Dice: {val_dice:.4f} (MA: {class_dices[0]:.4f}, HE: {class_dices[1]:.4f}, "
              f"EX: {class_dices[2]:.4f}, SE: {class_dices[3]:.4f}) | LR: {current_lr:.1e}")

        if val_dice > best_macro_dice:
            best_macro_dice = val_dice
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_macro_dice': best_macro_dice,
                'class_dices': class_dices,
                'image_size': 512,
                'num_classes': 4
            }
            torch.save(checkpoint, best_model_path)
            print(f"  --> Saved NEW BEST CHECKPOINT to {best_model_path} (Macro Dice: {best_macro_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60.0:.2f} min). Best epoch: {best_epoch} (Dice: {best_macro_dice:.4f})")

    # Save final model
    final_checkpoint = {
        'epoch': len(history),
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_macro_dice': best_macro_dice,
        'image_size': 512,
        'num_classes': 4
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"Saved FINAL CHECKPOINT to {final_model_path}")

    # Plot training loss & validation Dice curves
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(RESULT_DIR, "v3_training_history.csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df_hist['epoch'], df_hist['train_loss'], color='#2563eb', lw=2, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('V3 Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df_hist['epoch'], df_hist['val_macro_dice'], color='#059669', lw=2, label='Val Macro Dice')
    plt.axvline(best_epoch, color='#ef4444', linestyle=':', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Macro Dice')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "v3_training_curves.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
