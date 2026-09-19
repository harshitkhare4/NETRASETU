"""
src/evaluate_idrid_localization.py
Evaluation script for IDRiD Optic Disc and Fovea Center Keypoint Localization on the Locked Test Set (103 images).

Key requirements:
1. Uses models/NetraSetu_IDRiD_Localization_best.pth.
2. Evaluates the 103 official locked test images with zero tuning.
3. Computes:
   - Mean & median Euclidean pixel error
   - Standard deviation
   - Mean & median normalized error (error / image diagonal)
   - PCK@1%, PCK@2%, PCK@5%
   - Reports separately for Optic Disc, Fovea, and Combined.
4. Evaluates both Argmax and Subpixel Refinement.
5. Generates at least 10 qualitative overlay visualizations in results/localization/examples/.
6. Generates results/localization/worst_cases.csv (worst 10 OD and worst 10 Fovea cases).
7. Generates results/localization/test_results.txt and results/localization/localization_comparison.txt.
"""

import os
import sys
import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, r"C:\NetraSetu")
from src.train_idrid_localization import (
    KeypointResNet18,
    IDRiDLocalizationDataset,
    get_letterbox_params,
    processed_to_original_coordinate,
    extract_keypoint_from_heatmap
)

def evaluate_test_set():
    print("=" * 60)
    print("NETRASETU — IDRiD LOCALIZATION LOCKED TEST SET EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    best_model_path = r"C:\NetraSetu\models\NetraSetu_IDRiD_Localization_best.pth"
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    best_epoch = checkpoint.get('epoch', 'N/A')
    best_val_err = checkpoint.get('best_validation_error', 0.0)
    print(f"Loaded checkpoint from {best_model_path} (Trained Epoch {best_epoch}, Val Error: {best_val_err:.5f})")

    model = KeypointResNet18(out_channels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    manifest_path = r"C:\NetraSetu\data\IDRiD_Localization\manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    img_dir = r"C:\NetraSetu\data\IDRiD_Localization\test"

    test_dataset = IDRiDLocalizationDataset(
        manifest_df=manifest_df,
        img_dir=img_dir,
        split='test',
        is_train=False,
        heatmap_size=128,
        sigma=3.5
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Locked test dataset size: {len(test_dataset)} images")
    assert len(test_dataset) == 103, f"Expected 103 test images, found {len(test_dataset)}"

    results_dir = r"C:\NetraSetu\results\localization"
    examples_dir = os.path.join(results_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    records = []

    with torch.no_grad():
        for i, (images, heatmaps, meta) in enumerate(test_loader):
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred_heatmaps = model(images)

            pred_np = pred_heatmaps.float().cpu().numpy()[0]
            od_hm = pred_np[0]
            fov_hm = pred_np[1]

            # 1. Coordinate extraction with Subpixel refinement
            x_od_hm_sub, y_od_hm_sub = extract_keypoint_from_heatmap(od_hm, size=128, use_subpixel=True)
            x_fov_hm_sub, y_fov_hm_sub = extract_keypoint_from_heatmap(fov_hm, size=128, use_subpixel=True)

            # 2. Pure Argmax coordinate extraction
            x_od_hm_arg, y_od_hm_arg = extract_keypoint_from_heatmap(od_hm, size=128, use_subpixel=False)
            x_fov_hm_arg, y_fov_hm_arg = extract_keypoint_from_heatmap(fov_hm, size=128, use_subpixel=False)

            # Convert to 512x512 canvas
            scale_512 = 512.0 / 128.0
            x_od_512_sub, y_od_512_sub = x_od_hm_sub * scale_512, y_od_hm_sub * scale_512
            x_fov_512_sub, y_fov_512_sub = x_fov_hm_sub * scale_512, y_fov_hm_sub * scale_512

            x_od_512_arg, y_od_512_arg = x_od_hm_arg * scale_512, y_od_hm_arg * scale_512
            x_fov_512_arg, y_fov_512_arg = x_fov_hm_arg * scale_512, y_fov_hm_arg * scale_512

            # Map to original image resolution
            orig_w = float(meta['orig_w'][0])
            orig_h = float(meta['orig_h'][0])
            diag = math.sqrt(orig_w**2 + orig_h**2)

            od_x_pred_sub, od_y_pred_sub = processed_to_original_coordinate(x_od_512_sub, y_od_512_sub, orig_w, orig_h)
            fov_x_pred_sub, fov_y_pred_sub = processed_to_original_coordinate(x_fov_512_sub, y_fov_512_sub, orig_w, orig_h)

            od_x_pred_arg, od_y_pred_arg = processed_to_original_coordinate(x_od_512_arg, y_od_512_arg, orig_w, orig_h)
            fov_x_pred_arg, fov_y_pred_arg = processed_to_original_coordinate(x_fov_512_arg, y_fov_512_arg, orig_w, orig_h)

            gt_od_x = float(meta['od_x_orig'][0])
            gt_od_y = float(meta['od_y_orig'][0])
            gt_fov_x = float(meta['fovea_x_orig'][0])
            gt_fov_y = float(meta['fovea_y_orig'][0])
            img_name = str(meta['image_name'][0])

            # Errors Subpixel
            err_od_px_sub = math.hypot(od_x_pred_sub - gt_od_x, od_y_pred_sub - gt_od_y)
            err_fov_px_sub = math.hypot(fov_x_pred_sub - gt_fov_x, fov_y_pred_sub - gt_fov_y)

            # Errors Argmax
            err_od_px_arg = math.hypot(od_x_pred_arg - gt_od_x, od_y_pred_arg - gt_od_y)
            err_fov_px_arg = math.hypot(fov_x_pred_arg - gt_fov_x, fov_y_pred_arg - gt_fov_y)

            records.append({
                'image_name': img_name,
                'orig_w': orig_w,
                'orig_h': orig_h,
                'diag': diag,
                'gt_od_x': gt_od_x,
                'gt_od_y': gt_od_y,
                'gt_fov_x': gt_fov_x,
                'gt_fov_y': gt_fov_y,
                'pred_od_x_sub': od_x_pred_sub,
                'pred_od_y_sub': od_y_pred_sub,
                'pred_fov_x_sub': fov_x_pred_sub,
                'pred_fov_y_sub': fov_y_pred_sub,
                'err_od_px_sub': err_od_px_sub,
                'err_fov_px_sub': err_fov_px_sub,
                'norm_od_err_sub': err_od_px_sub / diag,
                'norm_fov_err_sub': err_fov_px_sub / diag,
                'pred_od_x_arg': od_x_pred_arg,
                'pred_od_y_arg': od_y_pred_arg,
                'pred_fov_x_arg': fov_x_pred_arg,
                'pred_fov_y_arg': fov_y_pred_arg,
                'err_od_px_arg': err_od_px_arg,
                'err_fov_px_arg': err_fov_px_arg,
                'norm_od_err_arg': err_od_px_arg / diag,
                'norm_fov_err_arg': err_fov_px_arg / diag,
            })

    df_res = pd.DataFrame(records)

    # ---------------------------------------------------------
    # COMPUTE METRICS (SUBPIXEL REFINEMENT AS PRIMARY)
    # ---------------------------------------------------------
    def summarize(px_errors, norm_errors):
        arr_px = np.array(px_errors)
        arr_norm = np.array(norm_errors)
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

    od_sub = summarize(df_res['err_od_px_sub'], df_res['norm_od_err_sub'])
    fov_sub = summarize(df_res['err_fov_px_sub'], df_res['norm_fov_err_sub'])
    comb_norm_sub = (od_sub['mean_norm'] + fov_sub['mean_norm']) / 2.0

    od_arg = summarize(df_res['err_od_px_arg'], df_res['norm_od_err_arg'])
    fov_arg = summarize(df_res['err_fov_px_arg'], df_res['norm_fov_err_arg'])
    comb_norm_arg = (od_arg['mean_norm'] + fov_arg['mean_norm']) / 2.0

    print("\nLOCKED TEST SET RESULTS (SUBPIXEL REFINEMENT - PRIMARY):")
    print(f"  Combined Mean Normalized Error: {comb_norm_sub:.5f} ({comb_norm_sub*100:.3f}% of diag)")
    print(f"  OPTIC DISC: Mean={od_sub['mean_px']:.2f}px, Median={od_sub['median_px']:.2f}px, Std={od_sub['std_px']:.2f}px, MeanNorm={od_sub['mean_norm']:.5f}")
    print(f"              PCK@1%={od_sub['pck_1']:.2f}%, PCK@2%={od_sub['pck_2']:.2f}%, PCK@5%={od_sub['pck_5']:.2f}%")
    print(f"  FOVEA:      Mean={fov_sub['mean_px']:.2f}px, Median={fov_sub['median_px']:.2f}px, Std={fov_sub['std_px']:.2f}px, MeanNorm={fov_sub['mean_norm']:.5f}")
    print(f"              PCK@1%={fov_sub['pck_1']:.2f}%, PCK@2%={fov_sub['pck_2']:.2f}%, PCK@5%={fov_sub['pck_5']:.2f}%")

    print("\nLOCKED TEST SET RESULTS (ARGMAX ONLY):")
    print(f"  Combined Mean Normalized Error: {comb_norm_arg:.5f}")
    print(f"  OPTIC DISC: Mean={od_arg['mean_px']:.2f}px, Median={od_arg['median_px']:.2f}px, PCK@2%={od_arg['pck_2']:.2f}%")
    print(f"  FOVEA:      Mean={fov_arg['mean_px']:.2f}px, Median={fov_arg['median_px']:.2f}px, PCK@2%={fov_arg['pck_2']:.2f}%")

    # ---------------------------------------------------------
    # ERROR CASE ANALYSIS: WORST 10 CASES
    # ---------------------------------------------------------
    worst_od = df_res.sort_values(by='err_od_px_sub', ascending=False).head(10).copy()
    worst_od['landmark'] = 'Optic Disc'
    worst_od['ground_truth'] = worst_od.apply(lambda r: f"({r['gt_od_x']:.1f}, {r['gt_od_y']:.1f})", axis=1)
    worst_od['prediction'] = worst_od.apply(lambda r: f"({r['pred_od_x_sub']:.1f}, {r['pred_od_y_sub']:.1f})", axis=1)
    worst_od['pixel_error'] = worst_od['err_od_px_sub'].round(2)
    worst_od['normalized_error'] = worst_od['norm_od_err_sub'].round(5)

    worst_fov = df_res.sort_values(by='err_fov_px_sub', ascending=False).head(10).copy()
    worst_fov['landmark'] = 'Fovea'
    worst_fov['ground_truth'] = worst_fov.apply(lambda r: f"({r['gt_fov_x']:.1f}, {r['gt_fov_y']:.1f})", axis=1)
    worst_fov['prediction'] = worst_fov.apply(lambda r: f"({r['pred_fov_x_sub']:.1f}, {r['pred_fov_y_sub']:.1f})", axis=1)
    worst_fov['pixel_error'] = worst_fov['err_fov_px_sub'].round(2)
    worst_fov['normalized_error'] = worst_fov['norm_fov_err_sub'].round(5)

    worst_combined = pd.concat([
        worst_od[['image_name', 'landmark', 'ground_truth', 'prediction', 'pixel_error', 'normalized_error']],
        worst_fov[['image_name', 'landmark', 'ground_truth', 'prediction', 'pixel_error', 'normalized_error']]
    ], ignore_index=True)

    worst_cases_path = os.path.join(results_dir, "worst_cases.csv")
    worst_combined.to_csv(worst_cases_path, index=False)
    print(f"\nSaved worst 10 error cases report to {worst_cases_path}")

    # ---------------------------------------------------------
    # QUALITATIVE VISUALIZATIONS (>= 10 EXAMPLES)
    # ---------------------------------------------------------
    print("\nGenerating qualitative visualizations for 12 test images...")
    # Choose 12 representative examples (spread evenly across test set)
    sample_indices = np.linspace(0, len(df_res) - 1, 12, dtype=int)

    for rank, idx in enumerate(sample_indices, 1):
        row = df_res.iloc[idx]
        img_name = row['image_name']
        test_img_path = os.path.join(img_dir, f"{img_name}.jpg")

        # Load 512x512 processed test image
        img_bgr = cv2.imread(test_img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Scale original coordinates to 512x512 canvas for plotting
        scale, pad_x, pad_y = get_letterbox_params(int(row['orig_w']), int(row['orig_h']), 512, 512)

        def to_512(x, y):
            return int(round(x * scale + pad_x)), int(round(y * scale + pad_y))

        gt_od_512 = to_512(row['gt_od_x'], row['gt_od_y'])
        pred_od_512 = to_512(row['pred_od_x_sub'], row['pred_od_y_sub'])
        gt_fov_512 = to_512(row['gt_fov_x'], row['gt_fov_y'])
        pred_fov_512 = to_512(row['pred_fov_x_sub'], row['pred_fov_y_sub'])

        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        ax.imshow(img_rgb)

        # Ground Truth OD: Green circle
        ax.plot(gt_od_512[0], gt_od_512[1], marker='o', markersize=10, markeredgecolor='#22c55e', markerfacecolor='none', markeredgewidth=2.5, label='GT Optic Disc')
        ax.plot(gt_od_512[0], gt_od_512[1], marker='+', markersize=8, color='#22c55e', markeredgewidth=2.0)

        # Predicted OD: Cyan cross
        ax.plot(pred_od_512[0], pred_od_512[1], marker='x', markersize=10, color='#06b6d4', markeredgewidth=2.5, label='Pred Optic Disc')

        # Ground Truth Fovea: Yellow circle
        ax.plot(gt_fov_512[0], gt_fov_512[1], marker='o', markersize=10, markeredgecolor='#eab308', markerfacecolor='none', markeredgewidth=2.5, label='GT Fovea')
        ax.plot(gt_fov_512[0], gt_fov_512[1], marker='+', markersize=8, color='#eab308', markeredgewidth=2.0)

        # Predicted Fovea: Magenta cross
        ax.plot(pred_fov_512[0], pred_fov_512[1], marker='x', markersize=10, color='#d946ef', markeredgewidth=2.5, label='Pred Fovea')

        # Connecting error line for visualization
        ax.plot([gt_od_512[0], pred_od_512[0]], [gt_od_512[1], pred_od_512[1]], color='#06b6d4', linestyle=':', lw=1.5)
        ax.plot([gt_fov_512[0], pred_fov_512[0]], [gt_fov_512[1], pred_fov_512[1]], color='#d946ef', linestyle=':', lw=1.5)

        ax.set_title(f"{img_name} | OD Err: {row['err_od_px_sub']:.1f}px ({row['norm_od_err_sub']*100:.2f}%) | "
                     f"Fov Err: {row['err_fov_px_sub']:.1f}px ({row['norm_fov_err_sub']*100:.2f}%)", fontsize=9, pad=8)
        ax.axis('off')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True, fontsize=8)

        example_out = os.path.join(examples_dir, f"localization_example_{rank:02d}_{img_name}.png")
        plt.savefig(example_out, bbox_inches='tight')
        plt.close()

    print(f"Saved 12 qualitative overlay visualizations to {examples_dir}")

    # ---------------------------------------------------------
    # TEST RESULTS REPORT (test_results.txt)
    # ---------------------------------------------------------
    test_report_lines = [
        "=" * 60,
        "NETRASETU — IDRiD LOCALIZATION LOCKED TEST RESULTS",
        "=" * 60,
        "Model Checkpoint:       models/NetraSetu_IDRiD_Localization_best.pth",
        f"Checkpoint Best Epoch:  {best_epoch}",
        f"Evaluated Test Images:  103 images (strictly locked, never seen during training/tuning)",
        "Prediction Mode:        Local Subpixel Centroid Refinement (Deterministic)",
        "",
        "OPTIC DISC LOCALIZATION (103 TEST IMAGES):",
        f"  Mean Euclidean Error:       {od_sub['mean_px']:.2f} px",
        f"  Median Euclidean Error:     {od_sub['median_px']:.2f} px",
        f"  Standard Deviation:         {od_sub['std_px']:.2f} px",
        f"  Mean Normalized Error:      {od_sub['mean_norm']:.5f} ({od_sub['mean_norm']*100:.3f}% of image diagonal)",
        f"  Median Normalized Error:    {od_sub['median_norm']:.5f} ({od_sub['median_norm']*100:.3f}% of image diagonal)",
        f"  PCK @ 1% (<= 51.5 px):      {od_sub['pck_1']:.2f}%",
        f"  PCK @ 2% (<= 103.0 px):     {od_sub['pck_2']:.2f}%",
        f"  PCK @ 5% (<= 257.4 px):     {od_sub['pck_5']:.2f}%",
        "",
        "FOVEA CENTER LOCALIZATION (103 TEST IMAGES):",
        f"  Mean Euclidean Error:       {fov_sub['mean_px']:.2f} px",
        f"  Median Euclidean Error:     {fov_sub['median_px']:.2f} px",
        f"  Standard Deviation:         {fov_sub['std_px']:.2f} px",
        f"  Mean Normalized Error:      {fov_sub['mean_norm']:.5f} ({fov_sub['mean_norm']*100:.3f}% of image diagonal)",
        f"  Median Normalized Error:    {fov_sub['median_norm']:.5f} ({fov_sub['median_norm']*100:.3f}% of image diagonal)",
        f"  PCK @ 1% (<= 51.5 px):      {fov_sub['pck_1']:.2f}%",
        f"  PCK @ 2% (<= 103.0 px):     {fov_sub['pck_2']:.2f}%",
        f"  PCK @ 5% (<= 257.4 px):     {fov_sub['pck_5']:.2f}%",
        "",
        "COMBINED ANATOMICAL LOCALIZATION:",
        f"  Mean Combined Normalized Error: {comb_norm_sub:.5f} ({comb_norm_sub*100:.3f}% of diagonal)",
        f"  Mean Combined Pixel Error:      {(od_sub['mean_px'] + fov_sub['mean_px'])/2.0:.2f} px",
        "",
        "SECONDARY COMPARISON — PURE ARGMAX (WITHOUT SUBPIXEL):",
        f"  OD Mean Error:              {od_arg['mean_px']:.2f} px | OD PCK@2%: {od_arg['pck_2']:.2f}%",
        f"  Fovea Mean Error:           {fov_arg['mean_px']:.2f} px | Fovea PCK@2%: {fov_arg['pck_2']:.2f}%",
        f"  Combined Norm Error:        {comb_norm_arg:.5f}",
        "=" * 60
    ]

    test_report_path = os.path.join(results_dir, "test_results.txt")
    with open(test_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_report_lines))
    print(f"Saved test results to {test_report_path}")

    # ---------------------------------------------------------
    # THREE-SPLIT COMPARISON (localization_comparison.txt)
    # ---------------------------------------------------------
    val_df = pd.read_csv(os.path.join(results_dir, "validation_results.csv"))
    best_val_row = val_df[val_df['epoch'] == best_epoch].iloc[0]

    comp_lines = [
        "=" * 70,
        "NETRASETU — IDRiD LOCALIZATION THREE-SPLIT COMPARISON",
        "=" * 70,
        f"{'Metric':<30} | {'Validation Split (62)':<18} | {'Locked Test Set (103)':<18}",
        "-" * 70,
        f"{'OD Mean Pixel Error':<30} | {best_val_row['od_mean_px']:>15.2f} px | {od_sub['mean_px']:>15.2f} px",
        f"{'OD Median Pixel Error':<30} | {best_val_row['od_median_px']:>15.2f} px | {od_sub['median_px']:>15.2f} px",
        f"{'OD PCK @ 1%':<30} | {best_val_row['od_pck_1']:>17.2f}% | {od_sub['pck_1']:>17.2f}%",
        f"{'OD PCK @ 2%':<30} | {best_val_row['od_pck_2']:>17.2f}% | {od_sub['pck_2']:>17.2f}%",
        f"{'OD PCK @ 5%':<30} | {best_val_row['od_pck_5']:>17.2f}% | {od_sub['pck_5']:>17.2f}%",
        "-" * 70,
        f"{'Fovea Mean Pixel Error':<30} | {best_val_row['fov_mean_px']:>15.2f} px | {fov_sub['mean_px']:>15.2f} px",
        f"{'Fovea Median Pixel Error':<30} | {best_val_row['fov_median_px']:>15.2f} px | {fov_sub['median_px']:>15.2f} px",
        f"{'Fovea PCK @ 1%':<30} | {best_val_row['fov_pck_1']:>17.2f}% | {fov_sub['pck_1']:>17.2f}%",
        f"{'Fovea PCK @ 2%':<30} | {best_val_row['fov_pck_2']:>17.2f}% | {fov_sub['pck_2']:>17.2f}%",
        f"{'Fovea PCK @ 5%':<30} | {best_val_row['fov_pck_5']:>17.2f}% | {fov_sub['pck_5']:>17.2f}%",
        "-" * 70,
        f"{'Combined Mean Norm Error':<30} | {best_val_row['combined_norm_error']:>18.5f} | {comb_norm_sub:>18.5f}",
        f"{'Combined Mean Pixel Error':<30} | {(best_val_row['od_mean_px']+best_val_row['fov_mean_px'])/2.0:>15.2f} px | {(od_sub['mean_px']+fov_sub['mean_px'])/2.0:>15.2f} px",
        "=" * 70
    ]

    comp_report_path = os.path.join(results_dir, "localization_comparison.txt")
    with open(comp_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(comp_lines))
    print(f"Saved three-split comparison to {comp_report_path}")
    print("\n".join(comp_lines))

if __name__ == "__main__":
    evaluate_test_set()
