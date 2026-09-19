"""
src/snapshot_checkpoints.py
Records size, modification timestamp, and SHA-256 hash for all production and experimental checkpoints.
Saves to results/final_pipeline/production_baseline_before.txt.
"""

import os
import hashlib
import time

def create_snapshot(out_file=r"C:\NetraSetu\results\final_pipeline\production_baseline_before.txt"):
    dirs = [
        r"C:\NetraSetu\results\final_pipeline",
        r"C:\NetraSetu\results\final_pipeline\segmentation",
        r"C:\NetraSetu\results\final_pipeline\vessel_segmentation",
        r"C:\NetraSetu\results\final_pipeline\calibration",
        r"C:\NetraSetu\results\final_pipeline\explainability",
        r"C:\NetraSetu\results\final_pipeline\explainability\gradcam_only",
        r"C:\NetraSetu\results\final_pipeline\explainability\lesion_only",
        r"C:\NetraSetu\results\final_pipeline\explainability\landmarks_only",
        r"C:\NetraSetu\results\final_pipeline\explainability\vessel_only",
        r"C:\NetraSetu\results\final_pipeline\explainability\combined_evidence",
        r"C:\NetraSetu\results\final_pipeline\integration",
        r"C:\NetraSetu\results\final_pipeline\capacity",
        r"C:\NetraSetu\results\final_pipeline\qa",
        r"C:\NetraSetu\results\final_pipeline\reports"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    models_to_hash = [
        r"C:\NetraSetu\models\NetraSetu_ResNet50_best.pth",
        r"C:\NetraSetu\models\NetraSetu_ResNet50_final.pth",
        r"C:\NetraSetu\models\NetraSetu_ResNet50_combined_best.pth",
        r"C:\NetraSetu\models\NetraSetu_ResNet50_combined_final.pth",
        r"C:\NetraSetu\models\NetraSetu_ResNet50_combined_v2_best.pth",
        r"C:\NetraSetu\models\NetraSetu_ResNet50_combined_v2_final.pth",
        r"C:\NetraSetu\models\NetraSetu_IDRiD_Localization_best.pth",
        r"C:\NetraSetu\models\NetraSetu_IDRiD_Localization_final.pth",
        r"C:\NetraSetu\models\NetraSetu_IDRiD_UNet_best.pth",
        r"C:\NetraSetu\models\NetraSetu_IDRiD_UNet_V2_best.pth",
        r"C:\NetraSetu\models\NetraSetu_IDRiD_UNet_V2_final.pth"
    ]

    lines = [
        "=" * 80,
        "NETRASETU — PRODUCTION BASELINE & MODEL CHECKPOINT SAFETY SNAPSHOT",
        "=" * 80,
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    for path in models_to_hash:
        if os.path.exists(path):
            stat = os.stat(path)
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                while chunk := f.read(8192 * 1024):
                    h.update(chunk)
            sha256 = h.hexdigest()
            lines.append(f"FILE:     {os.path.basename(path)}")
            lines.append(f"  Path:     {path}")
            lines.append(f"  Size:     {stat.st_size} bytes")
            lines.append(f"  Modified: {mtime}")
            lines.append(f"  SHA-256:  {sha256}")
            lines.append("-" * 80)
        else:
            lines.append(f"FILE NOT FOUND: {path}")
            lines.append("-" * 80)

    report_text = "\n".join(lines)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Safety snapshot written to {out_file}")
    return report_text

if __name__ == '__main__':
    create_snapshot()
