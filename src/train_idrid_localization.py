"""
src/train_idrid_localization.py
Training pipeline for NetraSetu IDRiD Optic Disc and Fovea Keypoint Localization.

Architecture:
- ResNet-18 ImageNet pretrained encoder + lightweight decoder with skip connections.
- Output: 2 heatmaps (128x128) - Channel 0: Optic Disc, Channel 1: Fovea.
- Input: 512x512 letterboxed retinal fundus images.

Target:
- 2D Gaussian heatmaps with sigma ~3.5 px on 128x128 grid.

Augmentation (Train only):
- HorizontalFlip with coordinated x-inversion.
- Small affine rotation (+/- 10 degrees) with matching coordinate rotation.
- Brightness/Contrast jitter (+/- 15%).

Validation Checkpoint Selection:
- Strictly based on lowest Validation Combined Mean Normalized Euclidean Localization Error.
- Model checkpoints saved to:
  models/NetraSetu_IDRiD_Localization_best.pth
  models/NetraSetu_IDRiD_Localization_final.pth
"""

import os
import sys
import time
import math
import json
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# GEOMETRY & COORDINATE UTILS
# -------------------------------------------------------------

def get_letterbox_params(orig_w=4288, orig_h=2848, target_w=512, target_h=512):
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    return scale, pad_x, pad_y

def processed_to_original_coordinate(x_proc, y_proc, orig_w=4288, orig_h=2848, target_w=512, target_h=512):
    scale, pad_x, pad_y = get_letterbox_params(orig_w, orig_h, target_w, target_h)
    x_orig = (x_proc - pad_x) / scale
    y_orig = (y_proc - pad_y) / scale
    return float(x_orig), float(y_orig)

def generate_gaussian_heatmap(x_center, y_center, size=128, sigma=3.5):
    """
    Generates a 128x128 Gaussian heatmap centered at (x_center, y_center).
    """
    yy, xx = np.mgrid[0:size, 0:size]
    heatmap = np.exp(-((xx - x_center)**2 + (yy - y_center)**2) / (2.0 * sigma**2))
    return heatmap.astype(np.float32)

def extract_keypoint_from_heatmap(heatmap, size=128, window_radius=3, use_subpixel=True):
    """
    Extracts keypoint coordinate from a 128x128 heatmap.
    Supports argmax with optional subpixel weighted centroid.
    """
    flat_idx = np.argmax(heatmap)
    y_arg, x_arg = np.unravel_index(flat_idx, (size, size))

    if not use_subpixel:
        return float(x_arg), float(y_arg)

    y_min = max(0, y_arg - window_radius)
    y_max = min(size, y_arg + window_radius + 1)
    x_min = max(0, x_arg - window_radius)
    x_max = min(size, x_arg + window_radius + 1)

    sub_h = heatmap[y_min:y_max, x_min:x_max]
    sum_w = np.sum(sub_h)
    if sum_w < 1e-6:
        return float(x_arg), float(y_arg)

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    x_sub = np.sum(xx * sub_h) / sum_w
    y_sub = np.sum(yy * sub_h) / sum_w
    return float(x_sub), float(y_sub)

# -------------------------------------------------------------
# DATASET & AUGMENTATION
# -------------------------------------------------------------

class IDRiDLocalizationDataset(Dataset):
    def __init__(self, manifest_df, img_dir, split='train', is_train=True, heatmap_size=128, sigma=3.5):
        self.df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")

        # Load 512x512 RGB image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        od_x = float(row['od_x_processed'])
        od_y = float(row['od_y_processed'])
        fov_x = float(row['fovea_x_processed'])
        fov_y = float(row['fovea_y_processed'])

        # Augmentation for training split only
        if self.is_train:
            # 1. Horizontal Flip (p=0.5)
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                od_x = 511.0 - od_x
                fov_x = 511.0 - fov_x

            # 2. Small Rotation +/- 10 degrees (p=0.5)
            if random.random() < 0.5:
                angle = random.uniform(-10.0, 10.0)
                center = (255.5, 255.5)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                
                # Transform coordinates
                pt_od = M.dot(np.array([od_x, od_y, 1.0]))
                pt_fov = M.dot(np.array([fov_x, fov_y, 1.0]))
                od_x, od_y = pt_od[0], pt_od[1]
                fov_x, fov_y = pt_fov[0], pt_fov[1]

            # 3. Brightness/Contrast (+/- 15%) (p=0.5)
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15) # contrast
                beta = random.uniform(-20.0, 20.0) # brightness
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            # Clamp coordinates safely within bounds
            od_x = np.clip(od_x, 0.0, 511.0)
            od_y = np.clip(od_y, 0.0, 511.0)
            fov_x = np.clip(fov_x, 0.0, 511.0)
            fov_y = np.clip(fov_y, 0.0, 511.0)

        # Scale coordinates to heatmap resolution (128x128)
        scale_hm = self.heatmap_size / 512.0
        od_hm_x = od_x * scale_hm
        od_hm_y = od_y * scale_hm
        fov_hm_x = fov_x * scale_hm
        fov_hm_y = fov_y * scale_hm

        # Generate target Gaussian heatmaps
        hm_od = generate_gaussian_heatmap(od_hm_x, od_hm_y, size=self.heatmap_size, sigma=self.sigma)
        hm_fov = generate_gaussian_heatmap(fov_hm_x, fov_hm_y, size=self.heatmap_size, sigma=self.sigma)
        target_heatmaps = np.stack([hm_od, hm_fov], axis=0) # (2, 128, 128)

        # Normalize image to Tensor (C, H, W)
        norm_img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        tensor_hm = torch.from_numpy(target_heatmaps).float()

        meta = {
            'image_name': img_name,
            'orig_w': float(row['original_width']),
            'orig_h': float(row['original_height']),
            'od_x_orig': float(row['od_x_original']),
            'od_y_orig': float(row['od_y_original']),
            'fovea_x_orig': float(row['fovea_x_original']),
            'fovea_y_orig': float(row['fovea_y_original'])
        }

        return tensor_img, tensor_hm, meta

# -------------------------------------------------------------
# MODEL ARCHITECTURE
# -------------------------------------------------------------

class KeypointResNet18(nn.Module):
    def __init__(self, out_channels=2):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.init_block = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.maxpool = base.maxpool
        self.layer1 = base.layer1 # 128x128, 64ch
        self.layer2 = base.layer2 # 64x64, 128ch
        self.layer3 = base.layer3 # 32x32, 256ch
        self.layer4 = base.layer4 # 16x16, 512ch

        # Decoder to upsample back to 128x128
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.init_block(x)
        x1 = self.maxpool(x0)
        l1 = self.layer1(x1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        d3 = self.conv3(torch.cat([self.up4(l4), l3], dim=1))
        d2 = self.conv2(torch.cat([self.up3(d3), l2], dim=1))
        d1 = self.conv1(torch.cat([self.up2(d2), l1], dim=1))
        out = self.head(d1)
        return out

# -------------------------------------------------------------
# EVALUATION & METRICS COMPUTATION
# -------------------------------------------------------------

def evaluate_model(model, dataloader, device, heatmap_size=128, use_subpixel=True):
    model.eval()
    criterion = nn.MSELoss()
    total_val_loss = 0.0

    od_pixel_errors = []
    fov_pixel_errors = []
    od_norm_errors = []
    fov_norm_errors = []

    with torch.no_grad():
        for images, heatmaps, meta in dataloader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred_heatmaps = model(images)
                loss = criterion(pred_heatmaps, heatmaps)
            total_val_loss += loss.item() * images.size(0)

            pred_np = pred_heatmaps.float().cpu().numpy()
            bs = images.size(0)

            for b in range(bs):
                # Extract OD and Fovea coordinates on 128x128 heatmap
                od_hm_pred = pred_np[b, 0]
                fov_hm_pred = pred_np[b, 1]

                x_od_hm, y_od_hm = extract_keypoint_from_heatmap(od_hm_pred, size=heatmap_size, use_subpixel=use_subpixel)
                x_fov_hm, y_fov_hm = extract_keypoint_from_heatmap(fov_hm_pred, size=heatmap_size, use_subpixel=use_subpixel)

                # Scale to 512x512 processed canvas
                scale_to_512 = 512.0 / heatmap_size
                x_od_512 = x_od_hm * scale_to_512
                y_od_512 = y_od_hm * scale_to_512
                x_fov_512 = x_fov_hm * scale_to_512
                y_fov_512 = y_fov_hm * scale_to_512

                # Map back to original image resolution
                orig_w = float(meta['orig_w'][b])
                orig_h = float(meta['orig_h'][b])
                diag = math.sqrt(orig_w**2 + orig_h**2)

                x_od_orig_pred, y_od_orig_pred = processed_to_original_coordinate(x_od_512, y_od_512, orig_w, orig_h)
                x_fov_orig_pred, y_fov_orig_pred = processed_to_original_coordinate(x_fov_512, y_fov_512, orig_w, orig_h)

                gt_od_x = float(meta['od_x_orig'][b])
                gt_od_y = float(meta['od_y_orig'][b])
                gt_fov_x = float(meta['fovea_x_orig'][b])
                gt_fov_y = float(meta['fovea_y_orig'][b])

                # Calculate Euclidean pixel errors
                err_od_px = math.hypot(x_od_orig_pred - gt_od_x, y_od_orig_pred - gt_od_y)
                err_fov_px = math.hypot(x_fov_orig_pred - gt_fov_x, y_fov_orig_pred - gt_fov_y)

                od_pixel_errors.append(err_od_px)
                fov_pixel_errors.append(err_fov_px)
                od_norm_errors.append(err_od_px / diag)
                fov_norm_errors.append(err_fov_px / diag)

    avg_loss = total_val_loss / len(dataloader.dataset)

    # Compute aggregate statistics
    def calc_metrics(errors_px, errors_norm):
        arr_px = np.array(errors_px)
        arr_norm = np.array(errors_norm)
        return {
            'mean_px': float(np.mean(arr_px)),
            'median_px': float(np.median(arr_px)),
            'std_px': float(np.std(arr_px)),
            'mean_norm': float(np.mean(arr_norm)),
            'median_norm': float(np.median(arr_norm)),
            'pck_1': float(np.mean(arr_norm <= 0.01) * 100.0),
            'pck_2': float(np.mean(arr_norm <= 0.02) * 100.0),
            'pck_5': float(np.mean(arr_norm <= 0.05) * 100.0),
        }

    od_stats = calc_metrics(od_pixel_errors, od_norm_errors)
    fov_stats = calc_metrics(fov_pixel_errors, fov_norm_errors)
    combined_norm_error = (od_stats['mean_norm'] + fov_stats['mean_norm']) / 2.0

    return avg_loss, od_stats, fov_stats, combined_norm_error

# -------------------------------------------------------------
# MAIN TRAINING PIPELINE
# -------------------------------------------------------------

def main():
    print("=" * 60)
    print("NETRASETU — IDRiD OPTIC DISC & FOVEA LOCALIZATION TRAINING")
    print("=" * 60)

    # Reproducibility seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    manifest_path = r"C:\NetraSetu\data\IDRiD_Localization\manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    img_base = r"C:\NetraSetu\data\IDRiD_Localization"

    train_dataset = IDRiDLocalizationDataset(
        manifest_df=manifest_df,
        img_dir=os.path.join(img_base, "train"),
        split='train',
        is_train=True,
        heatmap_size=128,
        sigma=3.5
    )
    val_dataset = IDRiDLocalizationDataset(
        manifest_df=manifest_df,
        img_dir=os.path.join(img_base, "val"),
        split='val',
        is_train=False,
        heatmap_size=128,
        sigma=3.5
    )

    print(f"Dataset loaded: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    # Dataloaders
    batch_size = 8
    num_workers = 2 if os.name != 'nt' else 0 # 0 on Windows avoids multiprocess locks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))

    # Model
    model = KeypointResNet18(out_channels=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = 25
    early_stopping_patience = 6
    epochs_without_improvement = 0
    best_combined_norm_error = float('inf')
    best_epoch = -1

    models_dir = r"C:\NetraSetu\models"
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "NetraSetu_IDRiD_Localization_best.pth")
    final_model_path = os.path.join(models_dir, "NetraSetu_IDRiD_Localization_final.pth")

    results_dir = r"C:\NetraSetu\results\localization"
    os.makedirs(results_dir, exist_ok=True)

    history = []
    val_rows = []

    start_time = time.time()
    print("\nStarting training loop...")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0

        for images, heatmaps, _ in train_loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(images)
                loss = criterion(preds, heatmaps)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_dataset)

        # Validation evaluation
        val_loss, od_stats, fov_stats, combined_norm_err = evaluate_model(
            model, val_loader, device, heatmap_size=128, use_subpixel=True
        )

        scheduler.step(combined_norm_err)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start

        # Record validation history
        val_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'combined_norm_error': combined_norm_err,
            'od_mean_px': od_stats['mean_px'],
            'od_median_px': od_stats['median_px'],
            'od_mean_norm': od_stats['mean_norm'],
            'od_pck_1': od_stats['pck_1'],
            'od_pck_2': od_stats['pck_2'],
            'od_pck_5': od_stats['pck_5'],
            'fov_mean_px': fov_stats['mean_px'],
            'fov_median_px': fov_stats['median_px'],
            'fov_mean_norm': fov_stats['mean_norm'],
            'fov_pck_1': fov_stats['pck_1'],
            'fov_pck_2': fov_stats['pck_2'],
            'fov_pck_5': fov_stats['pck_5'],
            'lr': current_lr,
            'duration_s': epoch_duration
        }
        val_rows.append(val_row)

        print(f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_duration:.1f}s] | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Comb Norm Err: {combined_norm_err:.4f} | "
              f"OD Mean: {od_stats['mean_px']:.1f}px (PCK@2%: {od_stats['pck_2']:.1f}%) | "
              f"Fov Mean: {fov_stats['mean_px']:.1f}px (PCK@2%: {fov_stats['pck_2']:.1f}%) | "
              f"LR: {current_lr:.1e}")

        # Checkpoint based on Validation Combined Normalized Error
        if combined_norm_err < best_combined_norm_error:
            best_combined_norm_error = combined_norm_err
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_validation_error': best_combined_norm_error,
                'od_stats': od_stats,
                'fov_stats': fov_stats,
                'image_size': 512,
                'heatmap_size': 128,
                'coordinate_transform_config': {
                    'orig_w': 4288,
                    'orig_h': 2848,
                    'scale': min(512/4288, 512/2848),
                    'pad_x': 0.0,
                    'pad_y': (512 - int(round(2848 * min(512/4288, 512/2848)))) / 2.0
                }
            }
            torch.save(checkpoint, best_model_path)
            print(f"  --> Saved NEW BEST CHECKPOINT to {best_model_path} (Comb Norm Err: {best_combined_norm_error:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs.")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s ({total_training_time/60.0:.2f} min). Best epoch: {best_epoch}")

    # Save final model
    final_checkpoint = {
        'epoch': len(val_rows),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'last_validation_error': val_rows[-1]['combined_norm_error'],
        'best_epoch': best_epoch,
        'best_validation_error': best_combined_norm_error,
        'image_size': 512,
        'heatmap_size': 128,
        'coordinate_transform_config': {
            'orig_w': 4288,
            'orig_h': 2848,
            'scale': min(512/4288, 512/2848),
            'pad_x': 0.0,
            'pad_y': (512 - int(round(2848 * min(512/4288, 512/2848)))) / 2.0
        }
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"Saved FINAL CHECKPOINT to {final_model_path}")

    # Save validation results CSV
    val_df = pd.DataFrame(val_rows)
    val_csv_path = os.path.join(results_dir, "validation_results.csv")
    val_df.to_csv(val_csv_path, index=False)
    print(f"Saved validation metrics to {val_csv_path}")

    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(val_df['epoch'], val_df['train_loss'], label='Train Heatmap MSE Loss', color='#2563eb', lw=2)
    plt.plot(val_df['epoch'], val_df['val_loss'], label='Val Heatmap MSE Loss', color='#dc2626', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Heatmap Training & Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_df['epoch'], val_df['combined_norm_error'], label='Val Combined Norm Error', color='#7c3aed', lw=2)
    plt.plot(val_df['epoch'], val_df['od_mean_norm'], label='OD Mean Norm Error', color='#059669', linestyle='--', lw=1.5)
    plt.plot(val_df['epoch'], val_df['fov_mean_norm'], label='Fovea Mean Norm Error', color='#d97706', linestyle='--', lw=1.5)
    plt.axvline(best_epoch, color='#ef4444', linestyle=':', label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Error (error / diag)')
    plt.title('Validation Localization Error')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Saved training curves to {curves_path}")

    # Save training_results.txt
    best_row = val_df[val_df['epoch'] == best_epoch].iloc[0]
    report_lines = [
        "=" * 60,
        "NETRASETU — IDRiD LOCALIZATION TRAINING REPORT",
        "=" * 60,
        f"Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset:                IDRiD C. Localization",
        f"Train Development Pool: 413 images",
        f"Training Split:         351 images (85%)",
        f"Validation Split:       62 images (15%)",
        f"Locked Test Set:        103 images (strictly withheld)",
        "",
        "PREPROCESSING & GEOMETRY:",
        f"Original Resolution:    4288 x 2848",
        f"Processed Resolution:   512 x 512 (letterboxed with aspect ratio preserved)",
        f"Letterbox Scale:        {min(512/4288, 512/2848):.6f}",
        f"Letterbox Padding:      pad_x = 0.0 px, pad_y = {(512 - int(round(2848 * min(512/4288, 512/2848)))) / 2.0:.1f} px",
        f"Heatmap Resolution:     128 x 128 (Gaussian sigma = 3.5 px)",
        "",
        "AUGMENTATION (TRAIN ONLY):",
        "  - HorizontalFlip (p=0.5, coordinated x-coordinate inversion)",
        "  - Small Rotation (+/- 10 deg, p=0.5, affine transformation matching coordinates)",
        "  - Brightness/Contrast (+/- 15%, p=0.5)",
        "",
        "ARCHITECTURE & HYPERPARAMETERS:",
        "Architecture:           KeypointResNet18 (ImageNet Pretrained Encoder + Bilinear Skip Decoder)",
        f"Loss Function:          MSE Heatmap Loss",
        f"Optimizer:              AdamW (lr=3e-4, weight_decay=1e-4)",
        f"Scheduler:              ReduceLROnPlateau (mode=min, factor=0.5, patience=3)",
        f"Batch Size:             {batch_size}",
        f"Max Epochs:             {num_epochs}",
        f"Early Stopping:         Patience = {early_stopping_patience}",
        f"Hardware / AMP:         {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}) / PyTorch AMP enabled",
        f"Total Training Time:    {total_training_time:.1f}s ({total_training_time/60.0:.2f} min)",
        "",
        "BEST VALIDATION CHECKPOINT PERFORMANCE:",
        f"Best Epoch:             {best_epoch}",
        f"Train Loss:             {best_row['train_loss']:.6f}",
        f"Validation Loss:        {best_row['val_loss']:.6f}",
        f"Combined Norm Error:    {best_row['combined_norm_error']:.5f} ({best_row['combined_norm_error']*100:.3f}% of diagonal)",
        "",
        "VALIDATION OPTIC DISC METRICS (AT BEST EPOCH):",
        f"  Mean Pixel Error:     {best_row['od_mean_px']:.2f} px",
        f"  Median Pixel Error:   {best_row['od_median_px']:.2f} px",
        f"  Mean Normalized Error:{best_row['od_mean_norm']:.5f}",
        f"  PCK @ 1% (<= 51.5 px):{best_row['od_pck_1']:.2f}%",
        f"  PCK @ 2% (<=103.0 px):{best_row['od_pck_2']:.2f}%",
        f"  PCK @ 5% (<=257.4 px):{best_row['od_pck_5']:.2f}%",
        "",
        "VALIDATION FOVEA METRICS (AT BEST EPOCH):",
        f"  Mean Pixel Error:     {best_row['fov_mean_px']:.2f} px",
        f"  Median Pixel Error:   {best_row['fov_median_px']:.2f} px",
        f"  Mean Normalized Error:{best_row['fov_mean_norm']:.5f}",
        f"  PCK @ 1% (<= 51.5 px):{best_row['fov_pck_1']:.2f}%",
        f"  PCK @ 2% (<=103.0 px):{best_row['fov_pck_2']:.2f}%",
        f"  PCK @ 5% (<=257.4 px):{best_row['fov_pck_5']:.2f}%",
        "=" * 60
    ]

    report_path = os.path.join(results_dir, "training_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved training results report to {report_path}")

if __name__ == "__main__":
    main()
