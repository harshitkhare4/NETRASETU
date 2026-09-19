"""
src/evaluate_drive_vessel.py
Evaluates DRIVE retinal vessel segmentation model.
- Evaluates held-out annotated validation set (Dice, IoU, Precision, Recall, Accuracy, Sensitivity, Specificity, ROC-AUC).
- Evaluates 20 locked test images with FOV masking.
- Generates at least 10 visual overlay examples in results/final_pipeline/vessel_segmentation/examples/.
- Saves report to results/final_pipeline/vessel_segmentation/test_results.txt.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, r"C:\NetraSetu")
from src.train_drive_vessel import VesselUNet, DRIVEDataset, val_tf, DRIVE_DIR, MODEL_DIR, RESULT_DIR

def evaluate_drive_vessel():
    print("=" * 60)
    print("NETRASETU — DRIVE RETINAL VESSEL EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(MODEL_DIR, "NetraSetu_DRIVE_Vessel_UNet_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = VesselUNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_epoch = checkpoint.get('epoch', 'N/A')
    best_dice = checkpoint.get('best_dice', 0.0)
    print(f"Loaded checkpoint from {model_path} (Epoch: {best_epoch}, Best Val Dice: {best_dice:.4f})")

    # 1. Quantitative Evaluation on Held-Out Validation Split with Ground Truth
    val_ds = DRIVEDataset(os.path.join(DRIVE_DIR, 'val'), transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    val_dices, val_ious, val_precs, val_recs = [], [], [], []
    val_accs, val_sens, val_specs, val_aucs = [], [], [], []

    with torch.no_grad():
        for imgs, masks, names in val_loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            gt = masks.numpy()[0, 0].astype(int)

            pred_bin = (probs >= 0.5).astype(int)

            # Metrics
            inter = np.sum(pred_bin * gt)
            dice = (2.0 * inter + 1e-6) / (np.sum(pred_bin) + np.sum(gt) + 1e-6)
            iou = (inter + 1e-6) / (np.sum(pred_bin) + np.sum(gt) - inter + 1e-6)
            acc = accuracy_score(gt.ravel(), pred_bin.ravel())
            prec = precision_score(gt.ravel(), pred_bin.ravel(), zero_division=0)
            rec = recall_score(gt.ravel(), pred_bin.ravel(), zero_division=0) # sensitivity

            # Specificity: TN / (TN + FP)
            tn = np.sum((1 - pred_bin) * (1 - gt))
            fp = np.sum(pred_bin * (1 - gt))
            spec = tn / (tn + fp + 1e-6)

            auc = roc_auc_score(gt.ravel(), probs.ravel())

            val_dices.append(dice)
            val_ious.append(iou)
            val_accs.append(acc)
            val_precs.append(prec)
            val_recs.append(rec)
            val_specs.append(spec)
            val_aucs.append(auc)

    print("\nHELD-OUT VALIDATION SET PERFORMANCE (GROUND TRUTH):")
    print(f"  Dice Score:   {np.mean(val_dices):.4f}")
    print(f"  IoU Score:    {np.mean(val_ious):.4f}")
    print(f"  Accuracy:     {np.mean(val_accs):.4f}")
    print(f"  Precision:    {np.mean(val_precs):.4f}")
    print(f"  Recall/Sens:  {np.mean(val_recs):.4f}")
    print(f"  Specificity:  {np.mean(val_specs):.4f}")
    print(f"  ROC-AUC:      {np.mean(val_aucs):.4f}")

    # 2. Locked Test Set Inference & Visualizations
    test_img_dir = os.path.join(DRIVE_DIR, 'test', 'images')
    test_fov_dir = os.path.join(DRIVE_DIR, 'test', 'fov_masks')
    examples_dir = os.path.join(RESULT_DIR, 'examples')
    os.makedirs(examples_dir, exist_ok=True)

    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
    print(f"\nEvaluating locked test set: {len(test_files)} images (01_test to 20_test)")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    test_records = []

    for rank, fname in enumerate(test_files, 1):
        img_p = os.path.join(test_img_dir, fname)
        fov_p = os.path.join(test_fov_dir, fname)

        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fov = cv2.imread(fov_p, cv2.IMREAD_GRAYSCALE)
        fov_bin = (fov > 127).astype(np.float32)

        # Normalize
        norm_img = (img_rgb / 255.0 - mean) / std
        tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(tensor_img)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

        # Apply FOV mask to avoid border artifacts
        probs = probs * fov_bin
        pred_vessels = (probs >= 0.5).astype(np.uint8) * 255

        vessel_density = float(np.sum(pred_vessels > 0) / max(np.sum(fov_bin > 0), 1))

        test_records.append({
            'filename': fname,
            'vessel_density_within_fov': round(vessel_density, 4),
            'mean_vessel_prob_within_fov': round(float(np.mean(probs[fov_bin > 0])), 4)
        })

        # Save qualitative visualization overlay for at least 12 images
        if rank <= 12:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
            axes[0].imshow(img_rgb)
            axes[0].set_title(f"{fname} (Original Fundus)", fontsize=10)
            axes[0].axis('off')

            axes[1].imshow(probs, cmap='inferno')
            axes[1].set_title("Predicted Vessel Probabilities", fontsize=10)
            axes[1].axis('off')

            # Create green vessel overlay on retinal fundus
            overlay = img_rgb.copy()
            vessel_mask_bool = (pred_vessels > 0)
            overlay[vessel_mask_bool] = [0, 255, 128] # vibrant green-cyan vessels
            blended = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)

            axes[2].imshow(blended)
            axes[2].set_title(f"Vessel Overlay (Density: {vessel_density*100:.1f}%)", fontsize=10)
            axes[2].axis('off')

            plt.tight_layout()
            out_img = os.path.join(examples_dir, f"vessel_overlay_{rank:02d}_{fname}")
            plt.savefig(out_img, bbox_inches='tight')
            plt.close()

    print(f"Saved 12 qualitative overlay visualizations in {examples_dir}")

    # Write test_results.txt
    report_lines = [
        "=" * 60,
        "NETRASETU — DRIVE RETINAL VESSEL SEGMENTATION REPORT",
        "=" * 60,
        "Dataset:              DRIVE (Digital Retinal Images for Vessel Extraction)",
        "Training Pool:        20 official training images (IDs 21-40)",
        "  Train Split:        17 images (85%)",
        "  Validation Split:   3 images (15%)",
        "Locked Test Set:      20 official test images (IDs 01-20)",
        "Model Checkpoint:     models/NetraSetu_DRIVE_Vessel_UNet_best.pth",
        f"Best Checkpoint Epoch:{best_epoch}",
        "",
        "HELD-OUT VALIDATION SET METRICS (WITH OFFICIAL GROUND TRUTH):",
        f"  Dice Score:         {np.mean(val_dices):.4f}",
        f"  IoU (Jaccard):      {np.mean(val_ious):.4f}",
        f"  Pixel Accuracy:     {np.mean(val_accs):.4f}",
        f"  Precision:          {np.mean(val_precs):.4f}",
        f"  Recall/Sensitivity: {np.mean(val_recs):.4f}",
        f"  Specificity:        {np.mean(val_specs):.4f}",
        f"  ROC-AUC:            {np.mean(val_aucs):.4f}",
        "",
        "LOCKED TEST SET (20 IMAGES) INFERENCE AUDIT:",
        f"  Evaluated images:   20 images (IDs 01-20)",
        f"  Mean Vessel Density:{np.mean([r['vessel_density_within_fov'] for r in test_records])*100:.2f}% of retinal FOV",
        f"  Mean Confidence:    {np.mean([r['mean_vessel_prob_within_fov'] for r in test_records]):.4f}",
        f"  Overlays generated: 12 high-resolution visual overlays in examples/",
        "",
        "RESEARCH STATUS:",
        "  - Dedicated standalone retinal vessel segmentation module.",
        "  - Evaluated on DRIVE benchmark. Not claimed as clinical validation on outside datasets.",
        "=" * 60
    ]

    report_path = os.path.join(RESULT_DIR, "test_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved results report to {report_path}")
    print("\n".join(report_lines))

if __name__ == '__main__':
    evaluate_drive_vessel()
