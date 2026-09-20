# NetraSetu — Render Production Model Deployment Guide

Smart India Hackathon 2026 — Problem Statement ID: 26038  
Project: **NetraSetu — Explainable AI for Diabetic Retinopathy Screening in Rural India**

---

## 1. Executive Summary

This document details the production model deployment architecture for NetraSetu on Render. It outlines resource sizing, memory profiling findings, secure model artifact delivery, and environment configuration.

---

## 2. Render Compute & Memory Sizing Architecture

### 2.1 Memory Profiling & Free Tier Sizing Analysis
Local and container profiling of the NetraSetu screening pipeline reveals the following memory allocations:

| Component | Resident Set Size (RSS) |
| :--- | :--- |
| Minimal Python Web Worker (Gunicorn + Flask) | ~25–35 MB |
| PyTorch Core Runtime (`import torch`) | ~400–520 MB |
| TorchVision + Computer Vision Pipeline (`torchvision`, `cv2`) | ~80–100 MB |
| Grad-CAM Explainability Engine (`pytorch_grad_cam`) | ~100–110 MB |
| ResNet-50 Production Weights (`NetraSetu_ResNet50_best.pth`, 94.4 MB) | ~95–140 MB |
| Working Inference Tensor Buffers (batch size 1, 512×512) | ~40–60 MB |
| **Total Peak Memory Footprint During Active Inference** | **~870 MB RAM** |

> [!WARNING]
> **Render Free Tier Limit (512 MB RAM):**  
> Render's Free web service tier provides an absolute ceiling of 512 MB RAM. When active PyTorch inference or eager model initialization takes place, memory consumption reaches ~870 MB, prompting the Linux kernel Out-Of-Memory (OOM) killer to terminate the worker via `SIGKILL`.  
> Therefore, **the Free 512 MB tier is insufficient for active PyTorch model inference**.

### 2.2 Recommended Production Service Plan
- **Recommended Instance Type**: Render Starter Instance (or higher)
- **CPU**: 1 vCPU
- **Memory**: **2 GB RAM (minimum recommended starting point for this deployment experiment)**
- **Operating System**: Linux (Ubuntu 22.04 LTS / Debian container)
- **Python Runtime**: Python 3.10.14
- **Note**: This is an engineering deployment choice to comfortably accommodate PyTorch memory allocation, not a clinical capacity claim.

---

## 3. Production Model Artifact Details

The production classifier weights are strictly locked and must match the verified integrity hash:

- **Checkpoint Filename**: `NetraSetu_ResNet50_best.pth`
- **Architecture**: Deep Residual Network (ResNet-50) with custom 5-class ICDR classifier head
- **Exact File Size**: `94,396,575` bytes (~94.4 MB)
- **Verified SHA-256 Checksum**:
  ```text
  258aa88fe0243c72f4c3f890515efb93da8a1b1bbc824b04f349e23966681ad6
  ```
- **Referable DR Threshold**: `0.24` (Locked)

> [!IMPORTANT]
> To prevent git bloat and comply with repository security policies, `models/*.pth` is permanently excluded from GitHub via `.gitignore`. Model weights must **never** be committed to Git.

---

## 4. Secure Model Delivery Strategies

### Strategy 1: Remote Artifact Streaming with Automated Checksum Verification (Recommended)
NetraSetu includes a built-in safe streaming model downloader (`src/download_model.py`).

1. Host the verified file `NetraSetu_ResNet50_best.pth` on a secure object store (e.g., GitHub Release Asset, AWS S3 presigned URL, Cloudflare R2, or Google Cloud Storage).
2. Configure the following environment variables in the **Render Dashboard** (do not commit secret tokens into `render.yaml`):
   ```bash
   NETRASETU_MODEL_URL=https://your-secure-storage.example.com/models/NetraSetu_ResNet50_best.pth
   NETRASETU_MODEL_SHA256=258aa88fe0243c72f4c3f890515efb93da8a1b1bbc824b04f349e23966681ad6
   ```
3. During startup or on first inference:
   - `src/download_model.py` downloads the file chunk-by-chunk to a temporary `.tmp` location.
   - Computes running SHA-256 during transit.
   - Verifies the hash against `258aa88fe0243c72f4c3f890515efb93da8a1b1bbc824b04f349e23966681ad6`.
   - Atomically moves the verified file to `models/NetraSetu_ResNet50_best.pth`.
   - If the checksum fails, the file is rejected and immediately deleted.

### Strategy 2: Render Persistent Disk
For persistent hosting across deployments without re-downloading:
1. Attach a Render Persistent Disk (e.g., 1 GB) mounted at `/opt/render/project/src/models`.
2. Place `NetraSetu_ResNet50_best.pth` directly on the disk.
3. Configure:
   ```bash
   NETRASETU_MODEL_PATH=/opt/render/project/src/models/NetraSetu_ResNet50_best.pth
   ```

---

## 5. Startup & Health Endpoint Behavior

1. **Lightweight Boot**:
   - The web server boots in < 1 second and consumes ~30 MB RAM.
   - Immediately binds to `0.0.0.0:$PORT` to satisfy Render's port check.
2. **GET `/health` Response**:
   - **Model Available**: HTTP 200
     ```json
     {
       "status": "ok",
       "service": "NetraSetu",
       "mode": "production",
       "model": "NetraSetu_ResNet50_best.pth",
       "referable_threshold": 0.24
     }
     ```
   - **Model Unavailable (Standby)**: HTTP 503
     ```json
     {
       "status": "degraded",
       "service": "NetraSetu",
       "mode": "production",
       "model": "NetraSetu_ResNet50_best.pth",
       "model_available": false,
       "message": "Production classifier model is unavailable"
     }
     ```
3. **Screening Endpoints (`/api/analyze`, `/api/sample/<id>`, `/api/analyze-camera`)**:
   - Guarded with controlled HTTP 503 status when model weights are missing.
   - Absolutely no dummy model fallback or random inference.
