"""
scratch/generate_project_inventory.py
Inspects actual project files and writes results/final_pipeline/qa/final_project_inventory.txt
"""

import os
import sys
import datetime

root = r"C:\NetraSetu"
inventory_lines = [
    "=" * 90,
    "NETRASETU — COMPLETE FINAL PROJECT INVENTORY",
    "=" * 90,
    f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ""
]

def audit_dir(rel_path, purpose_map):
    p = os.path.join(root, rel_path) if rel_path else root
    if not os.path.exists(p):
        inventory_lines.append(f"[{rel_path}] NOT FOUND\n")
        return
    inventory_lines.append(f"DIRECTORY: {rel_path or 'ROOT'}")
    inventory_lines.append("-" * 90)
    inventory_lines.append(f"{'Filename':<42} | {'Size':<10} | {'Modified':<19} | {'Purpose and Status'}")
    inventory_lines.append("-" * 90)
    
    entries = sorted(os.listdir(p))
    for e in entries:
        fp = os.path.join(p, e)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            if sz > 1024*1024:
                sz_str = f"{sz/(1024*1024):.1f} MB"
            elif sz > 1024:
                sz_str = f"{sz/1024:.1f} KB"
            else:
                sz_str = f"{sz} B"
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fp)).strftime('%Y-%m-%d %H:%M:%S')
            purpose, status = purpose_map.get(e, ("Project file", "Active"))
            inventory_lines.append(f"{e:<42} | {sz_str:<10} | {mtime:<19} | {purpose} [{status}]")
    inventory_lines.append("\n")

# 1. Models
model_purposes = {
    'NetraSetu_ResNet50_best.pth': ('Production baseline 5-class classifier', 'PROTECTED / LOCKED'),
    'NetraSetu_ResNet50_final.pth': ('Production baseline final epoch', 'PROTECTED / LOCKED'),
    'NetraSetu_ResNet50_combined_best.pth': ('Combined V1 APTOS+IDRiD classifier', 'Research Comparison'),
    'NetraSetu_ResNet50_combined_final.pth': ('Combined V1 final epoch', 'Research Comparison'),
    'NetraSetu_ResNet50_combined_v2_best.pth': ('Combined V2 quality-filtered classifier', 'Research Comparison'),
    'NetraSetu_ResNet50_combined_v2_final.pth': ('Combined V2 final epoch', 'Research Comparison'),
    'NetraSetu_IDRiD_Localization_best.pth': ('ResNet-18 OD & Fovea heatmap regression', 'Research Ready'),
    'NetraSetu_IDRiD_Localization_final.pth': ('Localization final epoch', 'Research Ready'),
    'NetraSetu_IDRiD_UNet_best.pth': ('IDRiD Lesion UNet V1', 'Historical Research'),
    'NetraSetu_IDRiD_UNet_V2_best.pth': ('IDRiD Lesion UNet V2', 'Historical Research'),
    'NetraSetu_IDRiD_UNet_V2_final.pth': ('IDRiD Lesion UNet V2 final', 'Historical Research'),
    'NetraSetu_IDRiD_UNet_V3_best.pth': ('IDRiD Lesion ResUNet V3 (Macro Dice 0.2965)', 'Research Deliverable'),
    'NetraSetu_IDRiD_UNet_V3_final.pth': ('IDRiD Lesion ResUNet V3 final', 'Research Deliverable'),
    'NetraSetu_DRIVE_Vessel_UNet_best.pth': ('DRIVE Retinal Vessel UNet (Dice 0.6780, Acc 93.7%)', 'Research Deliverable'),
    'NetraSetu_DRIVE_Vessel_UNet_final.pth': ('DRIVE Retinal Vessel UNet final', 'Research Deliverable'),
}
audit_dir('models', model_purposes)

# 2. Source scripts
src_purposes = {
    'inference.py': ('Core baseline inference functions and preprocessing', 'Production Core'),
    'inference_service.py': ('Flask inference service singleton and caching', 'Production Service'),
    'db.py': ('SQLite screening history, analytics and audit store', 'Production DB'),
    'explainability_engine.py': ('Unified 9-component explainability engine', 'Research Engine'),
    'research_pipeline.py': ('End-to-end research screening pipeline CLI', 'Research Pipeline'),
    'generate_explainable_report.py': ('Printable HTML and 1080p composite PNG report generator', 'Reporting Engine'),
    'export_explainability_layers.py': ('Batch 5-layer visual evidence exporter', 'Research Tool'),
    'capacity_simulator.py': ('100k-250k rural patient capacity and triage simulator', 'Planning Engine'),
    'calibrate_resnet50.py': ('Temperature scaling confidence calibrator (T=0.9215)', 'Calibration Tool'),
    'train_unet_v3.py': ('ResUNet V3 multi-class lesion training script', 'Training Code'),
    'optimize_segmentation_v3.py': ('Validation threshold optimizer for UNet V3', 'Evaluation Tool'),
    'evaluate_segmentation_v3.py': ('Official 27-image test evaluator for UNet V3', 'Evaluation Tool'),
    'prepare_drive_vessel.py': ('DRIVE dataset extractor and preprocessor', 'Preprocessing Tool'),
    'train_drive_vessel.py': ('Binary vessel segmentation UNet trainer', 'Training Code'),
    'evaluate_drive_vessel.py': ('Held-out DRIVE validation evaluator and overlay exporter', 'Evaluation Tool'),
    'verify_checkpoints_after.py': ('SHA-256 post-execution integrity auditor', 'Safety QA Tool'),
}
audit_dir('src', src_purposes)

# 3. Top-level files
top_purposes = {
    'app.py': ('Flask web application with /api/system_status and research mode', 'Production App'),
    'database.db': ('SQLite screening database', 'Production Store'),
    'README.md': ('Project technical documentation', 'Documentation'),
}
audit_dir('', top_purposes)

# 4. Results directories summary
inventory_lines.append('RESULTS & QA ARTIFACTS DIRECTORIES:')
inventory_lines.append('-' * 90)
results_base = os.path.join(root, 'results', 'final_pipeline')
for sub in sorted(os.listdir(results_base)):
    subp = os.path.join(results_base, sub)
    if os.path.isdir(subp):
        cnt = sum(len(files) for _, _, files in os.walk(subp))
        inventory_lines.append(f"results/final_pipeline/{sub:<24} | Directory ({cnt} files) | Active Artifacts")
    else:
        sz = os.path.getsize(subp)
        inventory_lines.append(f"results/final_pipeline/{sub:<24} | File ({sz} bytes) | Safety Snapshot")
inventory_lines.append('\n')

# 5. Backups
backup_base = os.path.join(root, 'backup')
inventory_lines.append('BACKUP DIRECTORIES:')
inventory_lines.append('-' * 90)
if os.path.exists(backup_base):
    for sub in sorted(os.listdir(backup_base)):
        subp = os.path.join(backup_base, sub)
        cnt = sum(len(files) for _, _, files in os.walk(subp)) if os.path.isdir(subp) else 1
        inventory_lines.append(f"backup/{sub:<36} | {cnt} items | Archived Baseline")
inventory_lines.append('\n' + '=' * 90)

out_txt = os.path.join(root, 'results', 'final_pipeline', 'qa', 'final_project_inventory.txt')
os.makedirs(os.path.dirname(out_txt), exist_ok=True)
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(inventory_lines))
print(f"Inventory successfully generated at: {out_txt}")
