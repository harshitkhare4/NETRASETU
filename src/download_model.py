"""
NetraSetu - Production Model Download & Integrity Verification
Smart India Hackathon 2026 - Problem Statement ID: 26038

Provides safe, streaming download and SHA-256 integrity verification
for the locked production ResNet-50 classifier checkpoint.
"""

import os
import sys
import hashlib
import logging
import urllib.request
import urllib.parse
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger("NetraSetuDownloader")

EXPECTED_PRODUCTION_SHA256 = "258aa88fe0243c72f4c3f890515efb93da8a1b1bbc824b04f349e23966681ad6"
EXPECTED_PRODUCTION_SIZE = 94396575
DEFAULT_MODEL_FILENAME = "NetraSetu_ResNet50_best.pth"


def compute_sha256(filepath: str, chunk_size: int = 1048576) -> str:
    """Computes the SHA-256 checksum of a file in 1 MB chunks."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest().lower()


def verify_model_file(filepath: str, expected_sha: Optional[str] = None) -> Tuple[bool, str]:
    """
    Verifies if a model file exists on disk and matches the expected SHA-256 checksum.
    Returns (is_valid, actual_sha_or_error).
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"

    expected = (expected_sha or os.getenv("NETRASETU_MODEL_SHA256", EXPECTED_PRODUCTION_SHA256)).strip().lower()
    try:
        actual_sha = compute_sha256(filepath)
        if actual_sha == expected:
            return True, actual_sha
        else:
            return False, f"Checksum mismatch: expected {expected[:12]}..., got {actual_sha[:12]}..."
    except Exception as e:
        return False, f"Error computing checksum: {e}"


def _sanitize_url(url: str) -> str:
    """Strips query parameters / access tokens from URLs for safe logging."""
    try:
        parsed = urllib.parse.urlparse(url)
        # Keep scheme, host, path only; strip query and fragment
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    except Exception:
        return "<sanitized-url>"


def download_model_from_url(
    url: str,
    target_path: str,
    expected_sha: Optional[str] = None,
    chunk_size: int = 65536
) -> bool:
    """
    Downloads model from URL using a streaming connection to a temporary file.
    Verifies SHA-256 before atomically replacing the target path.
    Never logs secrets or authentication tokens.
    """
    expected = (expected_sha or os.getenv("NETRASETU_MODEL_SHA256", EXPECTED_PRODUCTION_SHA256)).strip().lower()
    safe_url = _sanitize_url(url)
    logger.info(f"[NetraSetu Download] Initiating streaming model download from {safe_url}")

    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_dir = os.path.dirname(os.path.abspath(target_path))

    # Create temporary file in the same filesystem directory to ensure atomic os.replace
    temp_fd, temp_file_path = tempfile.mkstemp(prefix="model_download_", suffix=".tmp", dir=temp_dir)
    os.close(temp_fd)

    sha = hashlib.sha256()
    downloaded_bytes = 0

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NetraSetu-Model-Loader/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(temp_file_path, "wb") as out_f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    sha.update(chunk)
                    downloaded_bytes += len(chunk)

        computed_sha = sha.hexdigest().lower()
        logger.info(f"[NetraSetu Download] Downloaded {downloaded_bytes} bytes. Verifying SHA-256...")

        if computed_sha != expected:
            error_msg = (
                f"[NetraSetu Download] Checksum verification failed! "
                f"Expected: {expected}, Computed: {computed_sha}. Aborting."
            )
            logger.error(error_msg)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise ValueError(error_msg)

        # Atomic replacement
        os.replace(temp_file_path, target_path)
        logger.info(f"[NetraSetu Download] Model successfully verified and stored at: {target_path}")
        return True

    except Exception as e:
        logger.error(f"[NetraSetu Download] Failed to download model: {e}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
        raise


import threading

_download_lock = threading.Lock()


def ensure_model_available(target_path: Optional[str] = None) -> bool:
    """
    Ensures that the production model is present and verified on disk.
    If already present and valid SHA: returns True.
    If absent and NETRASETU_MODEL_URL is set: downloads and verifies.
    If absent and no URL set: returns False.
    Thread-safe with double-checked locking.
    """
    if target_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_env = os.getenv("NETRASETU_MODEL_PATH", "")
        if model_env:
            if not os.path.isabs(model_env):
                target_path = os.path.normpath(os.path.join(root, model_env))
            else:
                target_path = model_env
        else:
            target_path = os.path.join(root, "models", DEFAULT_MODEL_FILENAME)

    expected_sha = os.getenv("NETRASETU_MODEL_SHA256", EXPECTED_PRODUCTION_SHA256).strip().lower()

    # 1. Quick check without lock
    if os.path.exists(target_path):
        is_valid, msg = verify_model_file(target_path, expected_sha)
        if is_valid:
            return True

    # 2. Check if download URL is configured
    model_url = os.getenv("NETRASETU_MODEL_URL", "").strip()
    if not model_url:
        logger.warning(
            f"[NetraSetu] Model checkpoint not found at '{target_path}' "
            "and NETRASETU_MODEL_URL is not configured."
        )
        return False

    # 3. Synchronized download
    with _download_lock:
        # Double check if another thread completed it
        if os.path.exists(target_path):
            is_valid, msg = verify_model_file(target_path, expected_sha)
            if is_valid:
                return True

        try:
            return download_model_from_url(model_url, target_path, expected_sha)
        except Exception as e:
            logger.error(f"[NetraSetu] Automated model retrieval failed: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    success = ensure_model_available(dest)
    sys.exit(0 if success else 1)
