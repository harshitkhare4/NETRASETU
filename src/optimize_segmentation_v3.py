"""
src/optimize_segmentation_v3.py
Optimizes per-lesion channel binarization thresholds on the VALIDATION set only.
Candidate range: 0.20 to 0.70, step 0.02.
Outputs frozen thresholds to results/final_pipeline/segmentation/v3_optimal_thresholds.json.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, r"C:\NetraSetu")
from src.train_unet_v3 import UNetV3, IDRiDLesionDatasetV3, val_transform, compute_metrics, VAL_IMG_DIR, VAL_MASK_DIR, MODEL_DIR, RESULT_DIR

def optimize_thresholds():
    print("=" * 60)
    print("NETRASETU — V3 SEGMENTATION THRESHOLD OPTIMIZATION (VALIDATION ONLY)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = os.path.join(MODEL_DIR, "NetraSetu_IDRiD_UNet_V3_best.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model not found: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model = UNetV3(num_classes=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_dataset = IDRiDLesionDatasetV3(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Loaded validation set: {len(val_dataset)} images")

    # Collect all predictions and ground truth
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs[0])
            all_targets.append(masks[0])

    candidate_thresholds = np.arange(0.20, 0.72, 0.02)
    lesion_names = ["Microaneurysm", "Hemorrhage", "Hard Exudate", "Soft Exudate"]
    optimal_thresholds = {}
    tuning_records = []

    print("\nTuning thresholds per lesion class on validation split:")
    for c, name in enumerate(lesion_names):
        best_th = 0.50
        best_dice = -1.0
        best_metrics = {}

        for th in candidate_thresholds:
            th = round(float(th), 2)
            d_list, iou_list, p_list, r_list = [], [], [], []

            for probs, targets in zip(all_probs, all_targets):
                d, iou, p, r = compute_metrics(probs[c], targets[c], threshold=th)
                d_list.append(d)
                iou_list.append(iou)
                p_list.append(p)
                r_list.append(r)

            mean_d = float(np.mean(d_list))
            mean_iou = float(np.mean(iou_list))
            mean_p = float(np.mean(p_list))
            mean_r = float(np.mean(r_list))

            tuning_records.append({
                'class': name,
                'threshold': th,
                'val_dice': mean_d,
                'val_iou': mean_iou,
                'val_precision': mean_p,
                'val_recall': mean_r
            })

            if mean_d > best_dice:
                best_dice = mean_d
                best_th = th
                best_metrics = {
                    'threshold': th,
                    'val_dice': mean_d,
                    'val_iou': mean_iou,
                    'val_precision': mean_p,
                    'val_recall': mean_r
                }

        optimal_thresholds[name] = best_metrics
        print(f"  {name:15s} -> Optimal Threshold: {best_th:.2f} | Val Dice: {best_dice:.4f} (IoU: {best_metrics['val_iou']:.4f}, Prec: {best_metrics['val_precision']:.4f}, Rec: {best_metrics['val_recall']:.4f})")

    # Save frozen optimal thresholds
    out_json = os.path.join(RESULT_DIR, "v3_optimal_thresholds.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"\nFrozen optimal thresholds saved to {out_json}")

    pd.DataFrame(tuning_records).to_csv(os.path.join(RESULT_DIR, "v3_threshold_tuning_grid.csv"), index=False)
    return optimal_thresholds

if __name__ == "__main__":
    optimize_thresholds()
