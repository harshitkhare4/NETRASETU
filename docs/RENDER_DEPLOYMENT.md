# NetraSetu — Render Deployment Guide

This guide details the step-by-step procedure to deploy the **NetraSetu** AI screening platform as a Web Service on [Render](https://render.com).

---

## 1. Architecture Overview

NetraSetu deploys as a standard Python Web Service on Render running the production WSGI server **Gunicorn** (`gunicorn app:app`).

```
                              ┌─────────────────────────────┐
                              │  External HTTP Monitor      │
                              │  (e.g., UptimeRobot)        │
                              └──────────────┬──────────────┘
                                             │ Periodic Ping
                                             ▼ (/health)
┌────────────────────────────────────────────────────────────────────────┐
│ Render Web Service Container (Linux / Python 3.10)                     │
│                                                                        │
│   Gunicorn WSGI Server (app:app)                                       │
│   ├── GET  /health          Lightweight healthcheck (< 10 ms)          │
│   ├── GET  /api/health      Detailed system & telemetry state          │
│   ├── GET  /                Clinic Dashboard (HTML/CSS/JS)             │
│   └── POST /api/analyze     Quality Gate + ResNet-50 DR Grading (0.24) │
│                                                                        │
│   Mounted Storage / Disk                                               │
│   └── models/NetraSetu_ResNet50_best.pth (Production Baseline)         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Deployment on Render

### Step 1: Create a New Web Service
1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository: `https://github.com/harshitkhare4/NETRASETU`.

### Step 2: Configure Service Parameters

| Configuration Field | Value |
| :--- | :--- |
| **Name** | `netrasetu` (or custom name) |
| **Region** | Singapore / Frankfurt / Oregon (nearest to users) |
| **Branch** | `master` |
| **Root Directory** | *(Leave blank — project root)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Instance Type** | `Free` (or `Starter` for dedicated 24/7 memory) |

### Step 3: Health Check Path
Under **Advanced Settings**, configure:
- **Health Check Path:** `/health`

> **Note:** Do **not** use `/api/analyze` or `/` for health checking. `/health` is ultra-lightweight and returns immediately without running GPU/CPU tensor operations.

### Step 4: Environment Variables
Add the following environment variables in the Render Dashboard (**Environment** tab):

| Variable Name | Recommended Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.10.14` | Pins stable Python runtime. |
| `NETRASETU_RESEARCH_MODE` | `false` | **Enforces strict production baseline (APTOS ResNet-50, 0.24 threshold).** |
| `NETRASETU_MODEL_PATH` | `models/NetraSetu_ResNet50_best.pth` | Filepath to production checkpoint. |
| `SECRET_KEY` | *(Generate a random 32-char hex string)* | Flask session protection. |
| `FLASK_ENV` | `production` | Production environment flag. |

---

## 3. Model Weight Handling on Render

In accordance with repository best practices, large binary model checkpoints (`*.pth`) are **not stored in Git**.

### Deployment Weight Options:
1. **Render Persistent Disk (Recommended for Production):**
   - Attach a 1 GB persistent disk to your service mounted at `/mnt/models`.
   - Set environment variable: `NETRASETU_MODEL_PATH=/mnt/models/NetraSetu_ResNet50_best.pth`.
2. **Build-Time Download (via Build Command):**
   - If using cloud storage (e.g., S3, Google Cloud Storage, or private release assets), append a download command to the Build Command:
     ```bash
     pip install -r requirements.txt && curl -sSL "YOUR_SIGNED_WEIGHTS_URL" -o models/NetraSetu_ResNet50_best.pth
     ```

### Fail-Safe Model Check:
If the production checkpoint is absent at startup:
- The system logs an explicit warning: `WARNING: Primary classifier unavailable`.
- `/health` continues to return HTTP 200 with service metadata so the container remains healthy for inspection.
- `/api/health` indicates `"classifier_loaded": false` and `"status": "model_unavailable"`.
- Any call to `/api/analyze` immediately raises a clear error (`"ResNet-50 classifier is not loaded / model unavailable. Screening halted"`), **strictly preventing false or random screening results**.

---

## 4. Render Free-Tier Characteristics & External Monitoring

### Free Tier Sleep Policy
Render Free Web Services automatically **spin down into an idle state after 15 minutes of zero inbound traffic**.
- When an inbound request arrives, the service wakes up with a **cold-start delay** (~30–50 seconds while the container initializes and loads PyTorch into memory).
- After waking, subsequent requests execute with standard sub-second latency.

### External Monitoring with UptimeRobot
To keep the service responsive and minimize cold-start latency during demonstration hours, configure an external HTTP monitor:
1. Create a free monitor at [UptimeRobot](https://uptimerobot.com) or [BetterUptime](https://betteruptime.com).
2. Set Monitor Type: `HTTP(s)`.
3. Set Monitor URL: `https://<your-service-name>.onrender.com/health`.
4. Set Monitoring Interval: `Every 10 minutes` (or `14 minutes`).

> [!IMPORTANT]
> **No Internal Self-Pinging:** NetraSetu **does not** use internal background sleep/ping loops (`while True: requests.get(...)`). An idle or sleeping container cannot execute local loops to wake itself. Only an **external inbound HTTP request** from the internet triggers Render to resume a suspended instance.

---

## 5. Blueprint Deployment via `render.yaml`

A pre-configured Infrastructure-as-Code blueprint is provided in the repository at [`render.yaml`](../render.yaml).

To deploy via Blueprint:
1. In Render Dashboard, click **New +** → **Blueprint**.
2. Connect `harshitkhare4/NETRASETU`.
3. Render will automatically detect `render.yaml` and configure the service, healthcheck, build, and start commands.

---

## 6. Verification Checklist After Deployment

Once the service reports **Deploy Succeeded**:
- [ ] Verify `https://<service>.onrender.com/health` returns HTTP 200:
  ```json
  {
    "mode": "production",
    "model": "NetraSetu_ResNet50_best.pth",
    "referable_threshold": 0.24,
    "service": "NetraSetu",
    "status": "ok"
  }
  ```
- [ ] Open `https://<service>.onrender.com/` in your browser and verify the Light Healthcare UI renders.
- [ ] Verify `/api/health` returns device telemetry and `database_status: "connected"`.
- [ ] Verify `POST /api/sample/moderate-dr` executes and returns Grade 2 screening with Grad-CAM visualization.
