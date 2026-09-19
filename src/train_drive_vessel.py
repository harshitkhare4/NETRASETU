"""
src/train_drive_vessel.py
Training pipeline for DRIVE retinal blood vessel segmentation.
- Architecture: Lightweight Binary U-Net (1 output channel).
- Input: 512x512 RGB.
- Loss: Combined BCEWithLogitsLoss + Soft Dice Loss.
- Saves:
    models/NetraSetu_DRIVE_Vessel_UNet_best.pth
    models/NetraSetu_DRIVE_Vessel_UNet_final.pth
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------

PROJECT_ROOT = r"C:\NetraSetu"
DRIVE_DIR = os.path.join(PROJECT_ROOT, "data", "DRIVE")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "final_pipeline", "vessel_segmentation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------------------------------------
# DATASET
# -------------------------------------------------------------

class DRIVEDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.img_dir = os.path.join(split_dir, 'images')
        self.mask_dir = os.path.join(split_dir, 'masks')
        self.transform = transform
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask'].unsqueeze(0) # (1, H, W)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            img = (img - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask, fname

train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# -------------------------------------------------------------
# MODEL
# -------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class VesselUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

# -------------------------------------------------------------
# LOSS
# -------------------------------------------------------------

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.contiguous().view(-1)
        t = targets.contiguous().view(-1)
        intersection = (p * t).sum()
        dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return 1.0 - dice

# -------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------

def train_drive_vessel():
    print("=" * 60)
    print("NETRASETU — DRIVE RETINAL VESSEL UNET TRAINING")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = DRIVEDataset(os.path.join(DRIVE_DIR, 'train'), transform=train_tf)
    val_ds = DRIVEDataset(os.path.join(DRIVE_DIR, 'val'), transform=val_tf)

    print(f"Training set:   {len(train_ds)} images")
    print(f"Validation set: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = VesselUNet().to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.0]).to(device)) # vessels are ~12-14% of retinal area
    dice = BinaryDiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = 30
    best_dice = -1.0
    best_epoch = -1

    best_model_path = os.path.join(MODEL_DIR, "NetraSetu_DRIVE_Vessel_UNet_best.pth")
    final_model_path = os.path.join(MODEL_DIR, "NetraSetu_DRIVE_Vessel_UNet_final.pth")

    start_time = time.time()
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss = bce(logits, masks) + 1.2 * dice(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_dices = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                p = (probs >= 0.5).float()
                inter = (p * masks).sum().item()
                d = (2.0 * inter + 1e-6) / (p.sum().item() + masks.sum().item() + 1e-6)
                val_dices.append(d)

        mean_val_dice = float(np.mean(val_dices))
        scheduler.step(mean_val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_dice': mean_val_dice,
            'lr': current_lr
        })

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val Dice: {mean_val_dice:.4f} | LR: {current_lr:.1e}")

        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice
            }
            torch.save(checkpoint, best_model_path)
            print(f"  --> Saved NEW BEST to {best_model_path} (Dice: {best_dice:.4f})")

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_dice': best_dice
    }, final_model_path)

    total_time = time.time() - start_time
    print(f"\nVessel training finished in {total_time:.1f}s ({total_time/60.0:.2f} min). Best epoch: {best_epoch} (Val Dice: {best_dice:.4f})")

if __name__ == '__main__':
    train_drive_vessel()
