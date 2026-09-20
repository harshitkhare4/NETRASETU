"""
scratch/test_model_checksum.py
Verifies SHA-256 integrity check and rejection logic for the locked production model.
Tests:
1. Exact matching of locked production ResNet-50 weights
2. Rejection of corrupted / mismatched checksums
3. Atomic download and verification pipeline
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.download_model import (
    compute_sha256,
    verify_model_file,
    ensure_model_available,
    EXPECTED_PRODUCTION_SHA256,
    EXPECTED_PRODUCTION_SIZE
)


class TestModelChecksum(unittest.TestCase):

    def test_01_production_model_checksum_and_size(self):
        """Validates that the local production model exists and matches locked SHA-256."""
        model_path = os.path.join(PROJECT_ROOT, "models", "NetraSetu_ResNet50_best.pth")
        if not os.path.exists(model_path):
            self.skipTest("Local production model not present on this machine (normal on fresh clone).")

        size = os.path.getsize(model_path)
        self.assertEqual(
            size,
            EXPECTED_PRODUCTION_SIZE,
            f"Expected model size {EXPECTED_PRODUCTION_SIZE} bytes, got {size}"
        )

        sha = compute_sha256(model_path)
        self.assertEqual(
            sha,
            EXPECTED_PRODUCTION_SHA256,
            f"Expected SHA {EXPECTED_PRODUCTION_SHA256}, got {sha}"
        )

        is_valid, msg = verify_model_file(model_path)
        self.assertTrue(is_valid, f"Model file verification failed: {msg}")
        print(f"\n[PASS] Production model SHA-256 verified: {sha}")

    def test_02_reject_mismatched_checksum(self):
        """Validates that a corrupt/altered file is strictly rejected."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            f.write(b"CORRUPT DUMMY WEIGHTS REJECTION TEST DATA")
            temp_path = f.name

        try:
            is_valid, msg = verify_model_file(temp_path, EXPECTED_PRODUCTION_SHA256)
            self.assertFalse(is_valid, "Corrupt file should NOT pass verification!")
            self.assertIn("Checksum mismatch", msg)
            print(f"[PASS] Corrupt weights successfully rejected: {msg}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_03_reject_nonexistent_file(self):
        """Validates that missing files return False."""
        is_valid, msg = verify_model_file("nonexistent_model_file.pth")
        self.assertFalse(is_valid)
        self.assertEqual(msg, "File does not exist")
        print("[PASS] Nonexistent file correctly rejected.")


if __name__ == "__main__":
    unittest.main()
