"""
src/evaluate_segmentation_v3.py
Evaluates IDRiD Lesion Segmentation V3 on the locked 27-image test set using frozen thresholds.
Compares performance directly against the existing V2 benchmark.
Saves results to results/final_pipeline/segmentation/v3_test_results.txt.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, r"C:\NetraSetu")
from src.train_unet_v3 import UNetV3, IDRiDLesionDatasetV3, val_transform, compute_metrics, MODEL_DIR, RESULT_DIR

def evaluate_v3_test():
    print("=" * 60)
    print("NETRASETU — IDRiD LESION SEGMENTATION V3 LOCKED TEST EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = os.path.join(MODEL_DIR, "NetraSetu_IDRiD_UNet_V3_best.pth")
    thresholds_path = os.path.join(RESULT_DIR, "v3_optimal_thresholds.json")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {best_model_path}")
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Optimal thresholds not found: {thresholds_path}")

    with open(thresholds_path, "r", encoding="utf-8") as f:
        threshold_config = json.load(f)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model = UNetV3(num_classes=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_img_dir = r"C:\NetraSetu\data\IDRiD\processed\test\images"
    test_mask_dir = r"C:\NetraSetu\data\IDRiD\processed\test\masks"

    test_dataset = IDRiDLesionDatasetV3(test_img_dir, test_mask_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Loaded locked test set: {len(test_dataset)} images (strictly withheld)")
    assert len(test_dataset) == 27, f"Expected 27 test images, got {len(test_dataset)}"

    lesion_names = ["Microaneurysm", "Hemorrhage", "Hard Exudate", "Soft Exudate"]
    thresholds = [threshold_config[name]['threshold'] for name in lesion_names]
    print(f"Using frozen validation thresholds: {dict(zip(lesion_names, thresholds))}")

    all_dices = [[] for _ in range(4)]
    all_ious = [[] for _ in range(4)]
    all_precisions = [[] for _ in range(4)]
    all_recalls = [[] for _ in range(4)]

    per_image_results = []

    with torch.no_grad():
        for images, masks, names in test_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu()[0]
            targets = masks[0]
            img_name = names[0]

            row = {'image_name': img_name}
            for c, name in enumerate(lesion_names):
                d, iou, p, r = compute_metrics(probs[c], targets[c], threshold=thresholds[c])
                all_dices[c].append(d)
                all_ious[c].append(iou)
                all_precisions[c].append(p)
                all_recalls[c].append(r)
                row[f'{name}_Dice'] = d
                row[f'{name}_IoU'] = iou

            per_image_results.append(row)

    # Save per-image CSV
    pd.DataFrame(per_image_results).to_csv(os.path.join(RESULT_DIR, "v3_test_per_image_results.csv"), index=False)

    # Aggregate Test Metrics
    test_metrics = {}
    for c, name in enumerate(lesion_names):
        test_metrics[name] = {
            'threshold': thresholds[c],
            'dice': float(np.mean(all_dices[c])),
            'iou': float(np.mean(all_ious[c])),
            'precision': float(np.mean(all_precisions[c])),
            'recall': float(np.mean(all_recalls[c]))
        }

    macro_dice = float(np.mean([test_metrics[n]['dice'] for n in lesion_names]))
    macro_iou = float(np.mean([test_metrics[n]['iou'] for n in lesion_names]))

    # Existing V2 Benchmark Metrics for comparison
    v2_bench = {
        'Microaneurysm': {'dice': 0.0862, 'iou': 0.0471, 'precision': 0.1059, 'recall': 0.1051},
        'Hemorrhage':    {'dice': 0.3167, 'iou': 0.2057, 'precision': 0.4437, 'recall': 0.3371},
        'Hard Exudate':  {'dice': 0.4196, 'iou': 0.2947, 'precision': 0.4948, 'recall': 0.4885},
        'Soft Exudate':  {'dice': 0.3098, 'iou': 0.2626, 'precision': 0.2348, 'recall': 0.1447},
        'macro_dice': 0.2831,
        'macro_iou': 0.2025
    }

    # Decide preferred model
    v3_better = macro_dice > v2_bench['macro_dice']
    preferred = "V3 (NetraSetu_IDRiD_UNet_V3_best.pth)" if v3_better else "V2 (NetraSetu_IDRiD_UNet_V2_best.pth)"

    report_lines = [
        "=" * 65,
        "NETRASETU — IDRiD LESION SEGMENTATION V3 LOCKED TEST REPORT",
        "=" * 65,
        "Official Test Images: 27 images (strictly locked)",
        "Model Checkpoint:     models/NetraSetu_IDRiD_UNet_V3_best.pth",
        "Threshold Selection:  Validation set only (frozen)",
        "",
        "PER-LESION CLASS TEST RESULTS (V3):",
        "-" * 65
    ]

    for name in lesion_names:
        m = test_metrics[name]
        v2_m = v2_bench[name]
        delta_d = m['dice'] - v2_m['dice']
        report_lines.append(f"  {name.upper()}:")
        report_lines.append(f"    Threshold:   {m['threshold']:.2f}")
        report_lines.append(f"    Dice Score:  {m['dice']:.4f} (V2: {v2_m['dice']:.4f} | Delta: {delta_d:+.4f})")
        report_lines.append(f"    IoU Score:   {m['iou']:.4f} (V2: {v2_m['iou']:.4f})")
        report_lines.append(f"    Precision:   {m['precision']:.4f} (V2: {v2_m['precision']:.4f})")
        report_lines.append(f"    Recall:      {m['recall']:.4f} (V2: {v2_m['recall']:.4f})")
        report_lines.append("")

    report_lines.extend([
        "-" * 65,
        f"MACRO METRICS COMPARISON:",
        f"  V3 Macro Mean Dice: {macro_dice:.4f} | V2 Macro Mean Dice: {v2_bench['macro_dice']:.4f} | Delta: {macro_dice - v2_bench['macro_dice']:+.4f}",
        f"  V3 Macro Mean IoU:  {macro_iou:.4f} | V2 Macro Mean IoU:  {v2_bench['macro_iou']:.4f} | Delta: {macro_iou - v2_bench['macro_iou']:+.4f}",
        "",
        f"DECISION RULE ASSESSMENT:",
        f"  V3 Better than V2:               {'YES' if v3_better else 'NO'}",
        f"  Preferred Experimental Version:  {preferred}",
        "",
        "SCIENTIFIC / HONEST INTERPRETATION:",
        "  - Microaneurysm (MA) segmentation remains the most challenging lesion due to microscopic pixel diameter (<15px) and extreme rarity.",
        "  - Hard Exudates and Hemorrhages show strongest spatial coherence and Dice overlap.",
        "  - All segmentation outputs are designated for research explainability only (not clinical diagnostic use).",
        "=" * 65
    ])

    report_text = "\n".join(report_lines)
    test_rep_path = os.path.join(RESULT_DIR, "v3_test_results.txt")
    with open(test_rep_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save summary CSV
    summary_rows = []
    for name in lesion_names:
        summary_rows.append({
            'Lesion': name,
            'Threshold': test_metrics[name]['threshold'],
            'V3_Dice': test_metrics[name]['dice'],
            'V2_Dice': v2_bench[name]['dice'],
            'Dice_Delta': test_metrics[name]['dice'] - v2_bench[name]['dice'],
            'V3_IoU': test_metrics[name]['iou'],
            'V2_IoU': v2_bench[name]['iou'],
            'V3_Precision': test_metrics[name]['precision'],
            'V3_Recall': test_metrics[name]['recall']
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULT_DIR, "v3_confusion_or_summary.csv"), index=False)

    print(report_text)
    return test_metrics, preferred

if __name__ == "__main__":
    evaluate_v3_test()
