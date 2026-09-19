"""
src/research_pipeline.py
End-to-End Research Screening Inference Pipeline for NetraSetu.

Orchestrates:
1. Image Input & Pre-checks
2. Automated Quality Gate (GOOD / REVIEW / REJECT -> NEEDS RECAPTURE)
3. 5-Class DR Classification (NetraSetu_ResNet50_best.pth)
4. Confidence Calibration via Temperature Scaling (T = 0.9215)
5. Referable DR Risk Decision (P2 + P3 + P4 >= 0.24)
6. Grad-CAM Attention Heatmap
7. Anatomical Landmark Localization (Optic Disc & Fovea)
8. Multi-Class Lesion Segmentation (IDRiD UNet V3: MA, HE, Hard EX, Soft EX)
9. Retinal Vessel Network Segmentation (DRIVE Vessel UNet)
10. Multi-Modal Evidence Fusion Overlay & Clinical Screening Reports (HTML / PNG / JSON)

Usage:
  python src/research_pipeline.py --image path/to/image.jpg --output_dir results/screening/
  python src/research_pipeline.py --input_dir path/to/folder/ --output_dir results/screening/
"""

import os
import sys
import argparse
import json
import time
import cv2
import numpy as np

sys.path.insert(0, r"C:\NetraSetu")
from src.explainability_engine import ExplainabilityEngine
from src.generate_explainable_report import generate_report_for_image

class ResearchInferencePipeline:
    def __init__(self, device=None):
        self.engine = ExplainabilityEngine(device=device)

    def process_single_image(self, image_path, output_dir=None, patient_id=None):
        """
        Executes end-to-end inference on a single retinal fundus image.
        """
        t0 = time.time()
        fn = os.path.splitext(os.path.basename(image_path))[0]
        p_id = patient_id or f"NS-{fn}"
        
        is_idrid = ("IDRiD" in image_path)
        is_drive = ("DRIVE" in image_path or "drive" in fn.lower())

        print(f"\n========================================================")
        print(f" NetraSetu Research Pipeline: Processing [{p_id}]")
        print(f" Source: {image_path}")
        print(f"========================================================")

        # 1. Quality Assessment
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        quality = self.engine.assess_quality(img_bgr)
        print(f"Step 1: Quality Gate -> Status: {quality['status']} (Blur: {quality['blur_score']}, Contrast: {quality['contrast']}, Brightness: {quality['brightness']})")

        # Strict Quality Gate Enforcement
        if quality['status'] == "REJECT":
            print(f"  [HALTED] Image REJECTED by Quality Gate: {', '.join(quality['reasons'])}")
            print(f"  Action: Screening halted. Retinal recapture required.")
            res = {
                'patient_id': p_id,
                'status': 'NEEDS RECAPTURE',
                'quality': quality,
                'reasons': quality['reasons'],
                'clinical_action': 'Screening halted. Retinal recapture required.',
                'processing_time_sec': round(time.time() - t0, 3),
                'disclaimer': 'Research screening evaluation only. Not a clinical diagnosis.'
            }
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"{p_id}_recapture_notice.json"), 'w') as f:
                    json.dump(res, f, indent=2)
            return res

        # 2-9: Deep Model Inferences
        print("Step 2-9: Running Classification, Calibration, Grad-CAM, Localization, Lesion & Vessel Segmentation...")
        inf_res = self.engine.run_inference(image_path, is_idrid_domain=is_idrid, is_drive_domain=is_drive)

        elapsed = round(time.time() - t0, 3)
        print(f"Inference complete in {elapsed:.2f}s:")
        print(f"  - Severity Grade:       {inf_res['predicted_label']}")
        print(f"  - Calibrated Confidence: {inf_res['calibrated_confidence']*100:.1f}% (T=0.9215)")
        print(f"  - Referral Status:       {inf_res['referable_status']} (P_ref: {inf_res['referable_score']*100:.1f}%)")
        print(f"  - Anatomical Landmarks:  OD ({inf_res['optic_disc_prediction']['x']}, {inf_res['optic_disc_prediction']['y']}) | Fovea ({inf_res['fovea_prediction']['x']}, {inf_res['fovea_prediction']['y']})")
        print(f"  - Vessel Network:        Density: {inf_res['vessel_summary']['vessel_density']*100:.2f}%")
        
        detected_lesions = [f"{k} ({v['pixel_count']}px)" for k, v in inf_res['lesion_summary'].items() if v['status'] == 'detected']
        print(f"  - Detected Lesions:      {', '.join(detected_lesions) if detected_lesions else 'None detected'}")

        # 10. Generate Reports if output_dir provided
        report_record = None
        if output_dir:
            print("Step 10: Generating Printable HTML Screening Report & Composite Summary PNG...")
            report_record = generate_report_for_image(
                image_path=image_path,
                out_dir=output_dir,
                patient_id=p_id,
                engine=self.engine
            )

        inf_res['patient_id'] = p_id
        inf_res['processing_time_sec'] = elapsed
        inf_res['report_record'] = report_record
        return inf_res

    def process_batch(self, input_dir, output_dir, limit=None):
        os.makedirs(output_dir, exist_ok=True)
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
        
        if limit:
            image_files = image_files[:limit]

        print(f"\nBatch Screening Pipeline: {len(image_files)} images found in {input_dir}")
        batch_summary = []
        for i, img_p in enumerate(image_files):
            try:
                res = self.process_single_image(img_p, output_dir=output_dir, patient_id=f"BATCH-{i+1:03d}")
                summary_entry = {
                    'patient_id': res['patient_id'],
                    'file': os.path.basename(img_p),
                    'quality': res['quality']['status'],
                    'grade': res.get('predicted_label', 'RECAPTURED'),
                    'confidence': res.get('calibrated_confidence', 0.0),
                    'referable': res.get('referable_status', 'N/A'),
                    'time_sec': res['processing_time_sec']
                }
                batch_summary.append(summary_entry)
            except Exception as e:
                print(f"  [ERROR] Failed to process {img_p}: {e}")

        summary_file = os.path.join(output_dir, "batch_screening_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        print(f"\nBatch processing complete! Summary saved to {summary_file}")
        return batch_summary

def main():
    parser = argparse.ArgumentParser(description="NetraSetu End-to-End Research Screening Pipeline")
    parser.add_argument("--image", type=str, help="Path to single fundus image")
    parser.add_argument("--input_dir", type=str, help="Directory containing fundus images for batch screening")
    parser.add_argument("--output_dir", type=str, default=r"C:\NetraSetu\results\final_pipeline\screening_output", help="Directory to save reports and JSON records")
    parser.add_argument("--patient_id", type=str, default=None, help="Optional Patient ID")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process in batch mode")
    args = parser.parse_args()

    pipeline = ResearchInferencePipeline()

    if args.image:
        pipeline.process_single_image(args.image, output_dir=args.output_dir, patient_id=args.patient_id)
    elif args.input_dir:
        pipeline.process_batch(args.input_dir, output_dir=args.output_dir, limit=args.limit)
    else:
        print("Running pipeline self-test on representative image...")
        test_img = r"C:\NetraSetu\data\APTOS\raw\train_images\0024cdab0c1e.png"
        pipeline.process_single_image(test_img, output_dir=args.output_dir, patient_id="SELFTEST-001")

if __name__ == '__main__':
    main()
