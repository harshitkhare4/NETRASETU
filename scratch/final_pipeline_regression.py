"""
scratch/final_pipeline_regression.py
Regression test suite for the complete NetraSetu Final Research Pipeline.

Validates 6 critical test cases:
1. Valid GOOD APTOS image (end-to-end inference, calibrated DR grade, landmarks, lesions, vessel, reports)
2. Quality REJECTED image (must halt with NEEDS RECAPTURE, zero downstream crashes)
3. IDRiD Disease Grading image (in-domain localization & lesion evaluation)
4. IDRiD Localization image (validate OD/Fovea coordinates)
5. IDRiD Lesion Segmentation image (validate multi-class lesion outputs)
6. DRIVE Retinal Vessel image (validate binary vessel mask output and vessel density)

Saves summary to: results/final_pipeline/qa/regression_results.txt
"""

import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.insert(0, r"C:\NetraSetu")
from src.research_pipeline import ResearchInferencePipeline

def run_regression_suite():
    root = r"C:\NetraSetu"
    qa_dir = os.path.join(root, "results", "final_pipeline", "qa")
    out_dir = os.path.join(qa_dir, "regression_outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    pipeline = ResearchInferencePipeline()
    results = []
    
    print("\n========================================================")
    print(" NetraSetu Final Research Pipeline — Regression Test Suite")
    print("========================================================")

    # ----------------------------------------------------
    # Case 1: Valid GOOD APTOS image
    # ----------------------------------------------------
    print("\n--- Test Case 1: Valid GOOD APTOS Image ---")
    aptos_path = os.path.join(root, "data", "APTOS", "raw", "train_images", "002c21358ce6.png")
    if not os.path.exists(aptos_path):
        aptos_path = os.path.join(root, "data", "APTOS", "raw", "train_images", "0024cdab0c1e.png")
    
    t0 = time.time()
    res1 = pipeline.process_single_image(aptos_path, output_dir=out_dir, patient_id="REG-CASE-1-APTOS")
    pass1 = (
        res1.get('status') != 'NEEDS RECAPTURE' and
        res1.get('predicted_label') is not None and
        res1.get('calibrated_confidence') is not None and
        res1.get('optic_disc_prediction') is not None and
        res1.get('lesion_summary') is not None and
        res1.get('vessel_summary') is not None and
        os.path.exists(res1['report_record']['html_report']) and
        os.path.exists(res1['report_record']['composite_png'])
    )
    results.append({
        'case': 'Case 1: Valid GOOD APTOS image',
        'passed': pass1,
        'status': res1.get('status', 'SUCCESS'),
        'grade': res1.get('predicted_label'),
        'calib_conf': res1.get('calibrated_confidence'),
        'referable': res1.get('referable_status'),
        'elapsed_sec': round(time.time() - t0, 2)
    })
    print(f"Result Case 1: {'PASSED [OK]' if pass1 else 'FAILED [ERR]'}")

    # ----------------------------------------------------
    # Case 2: Quality REJECTED image
    # ----------------------------------------------------
    print("\n--- Test Case 2: Quality REJECTED Image (NEEDS RECAPTURE) ---")
    # Create an intentionally degraded severely blurred/dark image
    reject_path = os.path.join(out_dir, "synthetic_rejected_retina.jpg")
    blank_im = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(blank_im, (256, 256), 200, (5, 5, 5), -1) # Extremely dark and no contrast
    cv2.imwrite(reject_path, blank_im)
    
    t0 = time.time()
    res2 = pipeline.process_single_image(reject_path, output_dir=out_dir, patient_id="REG-CASE-2-REJECT")
    pass2 = (
        res2.get('status') == 'NEEDS RECAPTURE' and
        res2['quality']['status'] == 'REJECT' and
        len(res2.get('reasons', [])) > 0
    )
    results.append({
        'case': 'Case 2: Quality REJECTED image (Safety Gate)',
        'passed': pass2,
        'status': res2.get('status'),
        'reasons': res2.get('reasons'),
        'elapsed_sec': round(time.time() - t0, 2)
    })
    print(f"Result Case 2: {'PASSED [OK]' if pass2 else 'FAILED [ERR]'}")

    # ----------------------------------------------------
    # Case 3: IDRiD Disease Grading Image
    # ----------------------------------------------------
    print("\n--- Test Case 3: IDRiD Disease Grading Image ---")
    idrid_path = os.path.join(root, "data", "IDRiD_Grading", "B. Disease Grading", "1. Original Images", "a. Training Set", "IDRiD_002.jpg")
    t0 = time.time()
    res3 = pipeline.process_single_image(idrid_path, output_dir=out_dir, patient_id="REG-CASE-3-IDRID-GRADE")
    pass3 = (
        res3.get('status') != 'NEEDS RECAPTURE' and
        res3.get('predicted_label') is not None and
        res3['localization_meta']['domain_validated'] is True
    )
    results.append({
        'case': 'Case 3: IDRiD Disease Grading image',
        'passed': pass3,
        'status': res3.get('status', 'SUCCESS'),
        'grade': res3.get('predicted_label'),
        'domain_validated': res3['localization_meta']['domain_validated'],
        'elapsed_sec': round(time.time() - t0, 2)
    })
    print(f"Result Case 3: {'PASSED [OK]' if pass3 else 'FAILED [ERR]'}")

    # ----------------------------------------------------
    # Case 4: IDRiD Localization Image
    # ----------------------------------------------------
    print("\n--- Test Case 4: IDRiD Localization Image ---")
    loc_path = os.path.join(root, "data", "IDRiD_Grading", "B. Disease Grading", "1. Original Images", "a. Training Set", "IDRiD_003.jpg")
    t0 = time.time()
    res4 = pipeline.process_single_image(loc_path, output_dir=out_dir, patient_id="REG-CASE-4-IDRID-LOC")
    od = res4.get('optic_disc_prediction', {})
    fov = res4.get('fovea_prediction', {})
    pass4 = (
        od.get('x', 0) > 0 and od.get('y', 0) > 0 and
        fov.get('x', 0) > 0 and fov.get('y', 0) > 0
    )
    results.append({
        'case': 'Case 4: IDRiD Anatomical Landmark Localization',
        'passed': pass4,
        'status': 'SUCCESS' if pass4 else 'FAILED',
        'od': od,
        'fovea': fov,
        'elapsed_sec': round(time.time() - t0, 2)
    })
    print(f"Result Case 4: {'PASSED [OK]' if pass4 else 'FAILED [ERR]'}")

    # ----------------------------------------------------
    # Case 5: IDRiD Lesion Segmentation Image
    # ----------------------------------------------------
    print("\n--- Test Case 5: IDRiD Lesion Segmentation Image ---")
    seg_path = os.path.join(root, "data", "IDRiD_Grading", "B. Disease Grading", "1. Original Images", "a. Training Set", "IDRiD_004.jpg")
    t0 = time.time()
    res5 = pipeline.process_single_image(seg_path, output_dir=out_dir, patient_id="REG-CASE-5-IDRID-LESION")
    lesions = res5.get('lesion_summary', {})
    pass5 = (
        'Microaneurysm' in lesions and
        'Hemorrhage' in lesions and
        'Hard Exudate' in lesions and
        'Soft Exudate' in lesions
    )
    results.append({
        'case': 'Case 5: IDRiD UNet V3 Lesion Segmentation',
        'passed': pass5,
        'status': 'SUCCESS' if pass5 else 'FAILED',
        'lesions': {k: v['status'] for k, v in lesions.items()},
        'elapsed_sec': round(time.time() - t0, 2)
    })
    print(f"Result Case 5: {'PASSED [OK]' if pass5 else 'FAILED [ERR]'}")

    # ----------------------------------------------------
    # Case 6: DRIVE Retinal Vessel Image
    # ----------------------------------------------------
    print("\n--- Test Case 6: DRIVE Retinal Vessel Image ---")
    drive_path = os.path.join(root, "data", "DRIVE", "test", "images", "drive_01.png")
    t0 = time.time()
    res6 = pipeline.process_single_image(drive_path, output_dir=out_dir, patient_id="REG-CASE-6-DRIVE-VESSEL")
    vessel = res6.get('vessel_summary', {})
    pass6 = (
        vessel.get('available') is True and
        vessel.get('vessel_density', 0) > 0 and
        vessel.get('domain_validated') is True
    )
    results.append({
        'case': 'Case 6: DRIVE Retinal Vessel Segmentation',
        'passed': pass6,
        'status': 'SUCCESS' if pass6 else 'FAILED',
        'vessel_density': vessel.get('vessel_density'),
        'domain_validated': vessel.get('domain_validated'),
        'elapsed_sec': round(time.time() - t0, 2)
    })
    print(f"Result Case 6: {'PASSED [OK]' if pass6 else 'FAILED [ERR]'}")

    # ----------------------------------------------------
    # Write Regression Results Report
    # ----------------------------------------------------
    total_passed = sum(1 for r in results if r['passed'])
    total_cases = len(results)
    
    report_text = f"""================================================================================
NETRASETU RESEARCH PIPELINE — REGRESSION TEST SUITE RESULTS
Smart India Hackathon 2026 | Problem Statement 26038
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

OVERALL STATUS: {'ALL PASSED (6/6)' if total_passed == total_cases else f'{total_passed}/{total_cases} PASSED'}

SUMMARY TABLE:
--------------------------------------------------------------------------------
Case # | Description                                    | Status  | Time (s)
--------------------------------------------------------------------------------
1      | Valid GOOD APTOS Image (End-to-End)            | {'PASSED' if pass1 else 'FAILED'}  | {results[0]['elapsed_sec']}s
2      | Quality REJECTED Image (NEEDS RECAPTURE)       | {'PASSED' if pass2 else 'FAILED'}  | {results[1]['elapsed_sec']}s
3      | IDRiD Disease Grading Image (In-Domain)        | {'PASSED' if pass3 else 'FAILED'}  | {results[2]['elapsed_sec']}s
4      | IDRiD Anatomical Landmark Localization (OD/FOV)| {'PASSED' if pass4 else 'FAILED'}  | {results[3]['elapsed_sec']}s
5      | IDRiD UNet V3 Lesion Segmentation              | {'PASSED' if pass5 else 'FAILED'}  | {results[4]['elapsed_sec']}s
6      | DRIVE Retinal Vessel Segmentation              | {'PASSED' if pass6 else 'FAILED'}  | {results[5]['elapsed_sec']}s
--------------------------------------------------------------------------------

DETAILED EXECUTION NOTES:

Case 1 (GOOD APTOS):
- Predicted Grade: {res1.get('predicted_label')}
- Calibrated Confidence: {res1.get('calibrated_confidence', 0)*100:.1f}% (T=0.9215)
- Referral Status: {res1.get('referable_status')} (Risk: {res1.get('referable_score', 0)*100:.1f}%)
- Generated HTML Report: {res1['report_record']['html_report']}
- Generated Composite PNG: {res1['report_record']['composite_png']}

Case 2 (Safety Gate Rejection):
- Quality Gate Status: {res2['quality']['status']}
- Reasons for Halt: {', '.join(res2.get('reasons', []))}
- Downstream Inference Halted: YES (Zero deep model overhead)

Case 3 (IDRiD Grading):
- Predicted Grade: {res3.get('predicted_label')}
- In-Domain Localization Tag: {res3['localization_meta']['domain_validated']}

Case 4 (Landmark Localization):
- Optic Disc (x, y): ({od.get('x')}, {od.get('y')})
- Fovea (x, y): ({fov.get('x')}, {fov.get('y')})

Case 5 (Lesion Segmentation):
- Hemorrhages: {lesions.get('Hemorrhage', {}).get('status')} ({lesions.get('Hemorrhage', {}).get('pixel_count', 0)} px)
- Hard Exudates: {lesions.get('Hard Exudate', {}).get('status')} ({lesions.get('Hard Exudate', {}).get('pixel_count', 0)} px)
- Soft Exudates: {lesions.get('Soft Exudate', {}).get('status')} ({lesions.get('Soft Exudate', {}).get('pixel_count', 0)} px)
- Microaneurysms: {lesions.get('Microaneurysm', {}).get('status')} ({lesions.get('Microaneurysm', {}).get('pixel_count', 0)} px)

Case 6 (Vessel Segmentation):
- Retinal Vessel Density: {vessel.get('vessel_density', 0)*100:.2f}%
- Domain Validated: {vessel.get('domain_validated')}

================================================================================
End of Regression Report
================================================================================
"""
    results_txt_path = os.path.join(qa_dir, "regression_results.txt")
    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print(f"\nRegression test suite finished: {total_passed}/{total_cases} passed.")
    print(f"Saved results report to: {results_txt_path}")
    return results

if __name__ == '__main__':
    run_regression_suite()
