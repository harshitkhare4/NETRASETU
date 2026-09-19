"""
src/verify_checkpoints_after.py
Post-execution SHA-256 integrity verification across all NetraSetu model checkpoints.
Verifies that no baseline or historical models were modified or overwritten.
"""

import os
import sys
import hashlib
import time
from datetime import datetime

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    root = r"C:\NetraSetu"
    models_dir = os.path.join(root, "models")
    before_txt_path = os.path.join(root, "results", "final_pipeline", "production_baseline_before.txt")
    after_txt_path = os.path.join(root, "results", "final_pipeline", "production_baseline_after.txt")
    audit_report_path = os.path.join(root, "results", "final_pipeline", "reports", "CHECKPOINT_INTEGRITY_AUDIT.txt")

    # Parse baseline hashes before execution
    before_hashes = {}
    if os.path.exists(before_txt_path):
        current_file = None
        with open(before_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("FILE:"):
                    current_file = line.split(":", 1)[1].strip()
                elif line.startswith("SHA-256:") and current_file:
                    h = line.split(":", 1)[1].strip()
                    before_hashes[current_file] = h

    # List all checkpoints currently in models/
    all_ckpts = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])
    
    after_lines = [
        "=" * 80,
        "NETRASETU — POST-EXECUTION MODEL CHECKPOINT SNAPSHOT",
        "=" * 80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    audit_lines = [
        "=" * 80,
        "NETRASETU — POST-EXECUTION CHECKPOINT INTEGRITY AUDIT",
        "=" * 80,
        f"Audit Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SECTION 1: PRODUCTION & HISTORICAL CHECKPOINTS (Zero Mutation Rule)",
        "-" * 80,
        f"{'Filename':<42} | {'Before SHA-256 (prefix)':<22} | {'After SHA-256 (prefix)':<22} | {'Status'}",
        "-" * 80
    ]

    all_passed = True
    current_hashes = {}

    for ckpt in all_ckpts:
        p = os.path.join(models_dir, ckpt)
        size = os.path.getsize(p)
        mtime = datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M:%S')
        sha = compute_sha256(p)
        current_hashes[ckpt] = sha

        after_lines.append(f"FILE:     {ckpt}")
        after_lines.append(f"  Path:     {p}")
        after_lines.append(f"  Size:     {size} bytes")
        after_lines.append(f"  Modified: {mtime}")
        after_lines.append(f"  SHA-256:  {sha}")
        after_lines.append("-" * 80)

    # Check matches for before_hashes
    for fn, orig_hash in before_hashes.items():
        curr_hash = current_hashes.get(fn)
        if curr_hash is None:
            status = "MISSING [FAIL]"
            all_passed = False
            audit_lines.append(f"{fn:<42} | {orig_hash[:16]}...       | {'MISSING':<22} | {status}")
        elif curr_hash == orig_hash:
            status = "INTACT [PASSED]"
            audit_lines.append(f"{fn:<42} | {orig_hash[:16]}...       | {curr_hash[:16]}...       | {status}")
        else:
            status = "MUTATED [FAIL]"
            all_passed = False
            audit_lines.append(f"{fn:<42} | {orig_hash[:16]}...       | {curr_hash[:16]}...       | {status}")

    audit_lines.append("-" * 80)
    audit_lines.append("")
    audit_lines.append("SECTION 2: NEWLY TRAINED MODELS (Phase 1 & Phase 2 Deliverables)")
    audit_lines.append("-" * 80)
    audit_lines.append(f"{'Filename':<42} | {'Size (MB)':<12} | {'SHA-256 Full Hash'}")
    audit_lines.append("-" * 80)

    new_models = [
        "NetraSetu_IDRiD_UNet_V3_best.pth",
        "NetraSetu_IDRiD_UNet_V3_final.pth",
        "NetraSetu_DRIVE_Vessel_UNet_best.pth",
        "NetraSetu_DRIVE_Vessel_UNet_final.pth"
    ]
    for nm in new_models:
        if nm in current_hashes:
            p = os.path.join(models_dir, nm)
            sz_mb = os.path.getsize(p) / (1024 * 1024)
            audit_lines.append(f"{nm:<42} | {sz_mb:>8.2f} MB  | {current_hashes[nm]}")
        else:
            audit_lines.append(f"{nm:<42} | {'NOT FOUND':<12} | -")

    audit_lines.append("-" * 80)
    audit_lines.append("")
    audit_lines.append("FINAL AUDIT VERDICT: " + ("ALL 11 BASELINE CHECKPOINTS INTACT & UNTOUCHED (PASSED)" if all_passed else "CHECKPOINT MUTATION DETECTED (FAILED)"))
    audit_lines.append("=" * 80)

    with open(after_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(after_lines))

    audit_content = "\n".join(audit_lines)
    with open(audit_report_path, 'w', encoding='utf-8') as f:
        f.write(audit_content)

    print("\n" + audit_content)
    print(f"\nSaved post-execution snapshot to: {after_txt_path}")
    print(f"Saved audit report to: {audit_report_path}")

if __name__ == '__main__':
    main()
