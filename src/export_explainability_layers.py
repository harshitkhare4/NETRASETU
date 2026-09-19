"""
src/export_explainability_layers.py
Batch execution of multi-modal explainability layers across 12 diverse retinal images.
Generates:
- gradcam_only
- landmarks_only
- lesion_only
- vessel_only
- combined_evidence
"""

import os
import sys
import json
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, r"C:\NetraSetu")
from src.explainability_engine import ExplainabilityEngine

def draw_legend(img, entries, origin=(20, 30)):
    x, y = origin
    line_h = 28
    bg_w = 320
    bg_h = len(entries) * line_h + 16
    
    # Semi-transparent dark overlay for legend
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - 20), (x + bg_w, y + bg_h - 20), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x - 10, y - 20), (x + bg_w, y + bg_h - 20), (80, 80, 90), 1)

    for i, (label, color, shape) in enumerate(entries):
        cy = y + i * line_h
        if shape == 'circle':
            cv2.circle(img, (x + 10, cy), 8, color, -1)
            cv2.circle(img, (x + 10, cy), 8, (255, 255, 255), 1)
        elif shape == 'rect':
            cv2.rectangle(img, (x + 2, cy - 6), (x + 18, cy + 6), color, -1)
            cv2.rectangle(img, (x + 2, cy - 6), (x + 18, cy + 6), (255, 255, 255), 1)
        elif shape == 'cross':
            cv2.drawMarker(img, (x + 10, cy), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        cv2.putText(img, label, (x + 28, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 245), 1, cv2.LINE_AA)

def add_header_banner(img, title, subtitle=None):
    h, w = img.shape[:2]
    banner_h = 55 if subtitle else 38
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (15, 20, 25) # Dark slate
    
    cv2.putText(banner, title, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(banner, subtitle, (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 195), 1, cv2.LINE_AA)
        
    return np.vstack([banner, img])

def export_layers_for_image(engine, img_path, sample_id, dataset_name, out_dirs):
    print(f"\nProcessing [{sample_id}] from {dataset_name} ({os.path.basename(img_path)})...")
    
    # Run full inference
    is_idrid = ("IDRiD" in dataset_name)
    is_drive = ("DRIVE" in dataset_name)
    res = engine.run_inference(img_path, is_idrid_domain=is_idrid, is_drive_domain=is_drive)
    
    if res.get('status') == 'NEEDS RECAPTURE':
        print(f"  [WARN] Image {sample_id} rejected by Quality Gate: {res['reason']}")
        return None

    img_rgb = res['images']['original']
    orig_h, orig_w = img_rgb.shape[:2]
    
    grade_label = res['predicted_label']
    conf = res['calibrated_confidence'] * 100
    ref_status = res['referable_status']
    
    # 1. Grad-CAM Only
    cam_blend = res['images']['gradcam'].copy()
    cam_blend = add_header_banner(
        cam_blend,
        f"Grad-CAM Attention Map | {sample_id} ({dataset_name})",
        f"Predicted: {grade_label} (Conf: {conf:.1f}%) | {ref_status}"
    )
    cv2.imwrite(os.path.join(out_dirs['gradcam_only'], f"{sample_id}_gradcam.png"), cv2.cvtColor(cam_blend, cv2.COLOR_RGB2BGR))

    # 2. Landmarks Only
    landmarks_img = img_rgb.copy()
    od = res['optic_disc_prediction']
    fov = res['fovea_prediction']
    cv2.circle(landmarks_img, (int(od['x']), int(od['y'])), 40, (34, 197, 94), 3) # Green
    cv2.putText(landmarks_img, "OD", (int(od['x']) + 45, int(od['y']) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 197, 94), 2, cv2.LINE_AA)
    
    cv2.drawMarker(landmarks_img, (int(fov['x']), int(fov['y'])), (245, 158, 11), markerType=cv2.MARKER_CROSS, markerSize=32, thickness=3) # Orange
    cv2.putText(landmarks_img, "Fovea", (int(fov['x']) + 20, int(fov['y']) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 158, 11), 2, cv2.LINE_AA)
    
    draw_legend(landmarks_img, [
        ("Optic Disc Center (OD)", (34, 197, 94), "circle"),
        ("Fovea Centralis Center", (245, 158, 11), "cross"),
        (res['localization_meta']['disclaimer'], (180, 180, 180), "rect")
    ], origin=(25, 45))
    
    landmarks_img = add_header_banner(
        landmarks_img,
        f"Anatomical Landmark Localization | {sample_id} ({dataset_name})",
        f"OD: ({od['x']}, {od['y']}) | Fovea: ({fov['x']}, {fov['y']})"
    )
    cv2.imwrite(os.path.join(out_dirs['landmarks_only'], f"{sample_id}_landmarks.png"), cv2.cvtColor(landmarks_img, cv2.COLOR_RGB2BGR))

    # 3. Lesion Only
    lesion_img = img_rgb.copy()
    color_map = {
        'Microaneurysm': (234, 179, 8),    # Yellow
        'Hemorrhage': (239, 68, 68),       # Red
        'Hard Exudate': (6, 182, 212),     # Cyan
        'Soft Exudate': (217, 70, 239)     # Magenta
    }
    
    # Draw contour and mask fill
    if 'lesion_masks' in res['images']:
        for name, mask in res['images']['lesion_masks'].items():
            if np.sum(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(lesion_img, contours, -1, color_map[name], 2)
    
    legend_entries = [
        (f"Hemorrhage ({res['lesion_summary']['Hemorrhage']['status']} - {res['lesion_summary']['Hemorrhage']['pixel_count']} px)", (239, 68, 68), "rect"),
        (f"Hard Exudate ({res['lesion_summary']['Hard Exudate']['status']} - {res['lesion_summary']['Hard Exudate']['pixel_count']} px)", (6, 182, 212), "rect"),
        (f"Soft Exudate ({res['lesion_summary']['Soft Exudate']['status']} - {res['lesion_summary']['Soft Exudate']['pixel_count']} px)", (217, 70, 239), "rect"),
        (f"Microaneurysm ({res['lesion_summary']['Microaneurysm']['status']} - {res['lesion_summary']['Microaneurysm']['pixel_count']} px)", (234, 179, 8), "rect"),
    ]
    draw_legend(lesion_img, legend_entries, origin=(25, 45))
    
    lesion_img = add_header_banner(
        lesion_img,
        f"IDRiD UNet V3 Lesion Segmentation | {sample_id} ({dataset_name})",
        "Multi-class segmentation (MA, HE, Hard EX, Soft EX)"
    )
    cv2.imwrite(os.path.join(out_dirs['lesion_only'], f"{sample_id}_lesion.png"), cv2.cvtColor(lesion_img, cv2.COLOR_RGB2BGR))

    # 4. Vessel Only
    vessel_img = img_rgb.copy()
    vessel_mask = res['images']['vessel_mask']
    # Highlight vessels in electric emerald green
    vessel_overlay = vessel_img.copy()
    vessel_overlay[vessel_mask > 0] = [16, 185, 129] # Emerald
    cv2.addWeighted(vessel_overlay, 0.45, vessel_img, 0.55, 0, vessel_img)
    
    v_density = res['vessel_summary']['vessel_density'] * 100
    draw_legend(vessel_img, [
        (f"Retinal Blood Vessels (Density: {v_density:.1f}%)", (16, 185, 129), "rect"),
        (res['vessel_summary']['disclaimer'], (180, 180, 180), "rect")
    ], origin=(25, 45))
    
    vessel_img = add_header_banner(
        vessel_img,
        f"DRIVE Retinal Vessel Segmentation | {sample_id} ({dataset_name})",
        f"Vessel Network Density: {v_density:.2f}% | Model: NetraSetu DRIVE Vessel UNet"
    )
    cv2.imwrite(os.path.join(out_dirs['vessel_only'], f"{sample_id}_vessel.png"), cv2.cvtColor(vessel_img, cv2.COLOR_RGB2BGR))

    # 5. Combined Evidence
    combined_img = res['images']['combined_evidence'].copy()
    # Add combined legend
    all_legend = [
        (f"Grade: {grade_label} (Conf: {conf:.1f}%)", (255, 255, 255), "rect"),
        (f"Screening: {ref_status}", (239, 68, 68) if "REFERABLE" in ref_status else (34, 197, 94), "rect"),
        ("Optic Disc (OD)", (34, 197, 94), "circle"),
        ("Fovea Centralis", (245, 158, 11), "cross"),
        ("Hemorrhages (HE)", (239, 68, 68), "rect"),
        ("Hard Exudates (EX)", (6, 182, 212), "rect"),
        ("Soft Exudates (SE)", (217, 70, 239), "rect"),
        ("Microaneurysms (MA)", (234, 179, 8), "rect"),
        ("Background: Grad-CAM attention", (140, 140, 140), "rect")
    ]
    draw_legend(combined_img, all_legend, origin=(25, 45))
    
    combined_img = add_header_banner(
        combined_img,
        f"NetraSetu Unified Multi-Modal Explainability | {sample_id} ({dataset_name})",
        "Clinical Research AI Screening — Integrated Classification, Localization, Lesion & Vessel Evidence"
    )
    cv2.imwrite(os.path.join(out_dirs['combined_evidence'], f"{sample_id}_combined.png"), cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

    print(f"  [OK] Exported 5 layers for {sample_id}")
    return {
        'sample_id': sample_id,
        'dataset': dataset_name,
        'path': img_path,
        'grade': grade_label,
        'confidence': round(conf / 100.0, 4),
        'referable': ref_status,
        'od': od,
        'fovea': fov,
        'vessel_density': v_density,
        'lesions': res['lesion_summary']
    }

def main():
    root = r"C:\NetraSetu"
    base_out = os.path.join(root, "results", "final_pipeline", "explainability")
    out_dirs = {
        'gradcam_only': os.path.join(base_out, "gradcam_only"),
        'landmarks_only': os.path.join(base_out, "landmarks_only"),
        'lesion_only': os.path.join(base_out, "lesion_only"),
        'vessel_only': os.path.join(base_out, "vessel_only"),
        'combined_evidence': os.path.join(base_out, "combined_evidence"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    engine = ExplainabilityEngine()

    # Define 12 diverse test images: 4 APTOS, 4 IDRiD, 4 DRIVE
    samples = []

    # 1. APTOS (raw train images)
    aptos_dir = os.path.join(root, "data", "APTOS", "raw", "train_images")
    if os.path.exists(aptos_dir):
        files = [f for f in os.listdir(aptos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Filter files that pass quality
        aptos_count = 0
        for fn in files:
            p = os.path.join(aptos_dir, fn)
            im = cv2.imread(p)
            if im is not None and engine.assess_quality(im)['status'] in ['GOOD', 'REVIEW']:
                aptos_count += 1
                samples.append((f"APTOS_Sample_{aptos_count}", "APTOS", p))
                if aptos_count >= 4:
                    break

    # 2. IDRiD (Disease grading images)
    idrid_dir = os.path.join(root, "data", "IDRiD_Grading", "B. Disease Grading", "1. Original Images", "a. Training Set")
    if not os.path.exists(idrid_dir):
        idrid_dir = os.path.join(root, "data", "IDRiD_Grading", "B. Disease Grading", "1. Original Images", "b. Testing Set")
    if os.path.exists(idrid_dir):
        files = [f for f in os.listdir(idrid_dir) if f.lower().endswith(('.jpg', '.tif', '.png'))]
        idrid_count = 0
        for fn in files:
            p = os.path.join(idrid_dir, fn)
            im = cv2.imread(p)
            if im is not None and engine.assess_quality(im)['status'] in ['GOOD', 'REVIEW']:
                idrid_count += 1
                samples.append((f"IDRiD_Sample_{idrid_count}", "IDRiD", p))
                if idrid_count >= 4:
                    break

    # 3. DRIVE (Training / Test images)
    drive_dir = os.path.join(root, "data", "DRIVE", "test", "images")
    if not os.path.exists(drive_dir):
        drive_dir = os.path.join(root, "data", "DRIVE", "training", "images")
    if os.path.exists(drive_dir):
        files = [f for f in os.listdir(drive_dir) if f.lower().endswith(('.tif', '.png', '.jpg'))]
        drive_count = 0
        for fn in files:
            p = os.path.join(drive_dir, fn)
            drive_count += 1
            samples.append((f"DRIVE_Sample_{drive_count}", "DRIVE", p))
            if drive_count >= 4:
                break

    print(f"Selected {len(samples)} valid images across APTOS, IDRiD, and DRIVE.")

    results_meta = []
    for s_id, d_name, p in samples:
        res = export_layers_for_image(engine, p, s_id, d_name, out_dirs)
        if res:
            results_meta.append(res)

    summary_file = os.path.join(base_out, "explainability_samples_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_meta, f, indent=2)
    print(f"\nCompleted! Generated {len(results_meta)*5} visualization layers.")
    print(f"Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
