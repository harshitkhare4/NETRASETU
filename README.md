# NetraSetu (नेत्रसेतु)
### Explainable AI for Diabetic Retinopathy Screening in Rural India
**Smart India Hackathon 2026** | **Problem Statement ID: 26038**  
**Category:** Software | **Theme:** Healthcare & Biomedical Devices  
**Clinical Status:** AI-Assisted Research Screening Prototype — *Clinical Validation Not Performed*

---

## 1. Executive Overview

**NetraSetu** ("Bridge to Vision") is an end-to-end, safety-first clinical decision support platform designed to address the critical shortage of eye care specialists in rural and semi-urban India. Over 77 million people in India live with diabetes, yet the doctor-to-patient ratio in rural districts often exceeds 1:100,000. Diabetic Retinopathy (DR) is often asymptomatic in its early stages and only detected when vision loss is irreversible.

NetraSetu provides:
1. **Automated Image Quality Gating:** Detects motion blur, poor contrast, underexposure, and off-retina fields to issue immediate **NEEDS RECAPTURE** warnings while the patient is still present.
2. **Standard 5-Class DR Severity Staging:** Categorizes retinal fundus photos into Grade 0 (No DR), Grade 1 (Mild NPDR), Grade 2 (Moderate NPDR), Grade 3 (Severe NPDR), and Grade 4 (Proliferative DR) using an optimized ResNet-50 baseline.
3. **Sensitivity-Prioritized Referable Triage:** Employs a locked decision threshold ($P_{\text{ref}} \ge 0.24$) to maximize detection of sight-threatening disease.
4. **Post-Hoc Confidence Calibration:** Calibrates neural network outputs via Temperature Scaling ($T=0.9215$) to produce reliable probabilities for healthcare staff.
5. **Unified Multi-Modal Explainability Studio:** Combines Grad-CAM class activation maps, anatomical landmark localization (Optic Disc & Fovea), multi-class lesion segmentation (Hemorrhages, Hard/Soft Exudates, Microaneurysms), and retinal vessel tree density.
6. **Automated Clinical Screening Reports:** Generates self-contained printable HTML reports, 1080p composite visual summary cards, and machine-readable JSON records.
7. **Rural Capacity & Throughput Modeling:** Demonstrates that an entry-level workstation can screen 100,000+ patients annually with sub-second latency (325.2 ms).

---

## 2. System Architecture

```
[ Retinal Fundus Input (Upload / Camera / Sample) ]
                       │
                       ▼
         [ 1. Image Quality Gate ]
           - Laplacian Blur Variance (< 8.0 = REJECT)
           - Grayscale Contrast & Brightness Checks
           - Retinal Field of View (FOV) Ratio
           ├── Status: REJECT ──► [ INSTANT RECAPTURE ALERT ]
           │                      (Downstream deep inference halted safely)
           └── Status: GOOD / REVIEW
                       │
                       ▼
     [ 2. Preprocessing & Standardization ]
       - Circular Masking & Retinal Auto-Crop
       - Bilateral Filtering (Edge-preserving denoising)
       - CLAHE (Illumination contrast equalization in Lab space)
                       │
                       ▼
       [ 3. Deep Learning Inference Modules ]
       ├── A. 5-Class DR Classifier (ResNet-50 Baseline)
       │      └── Temperature Scaling Calibration (T = 0.9215)
       │      └── Binary Referable Triage Decision (P_ref >= 0.24)
       ├── B. Visual Attention (Grad-CAM on layer4[-1])
       ├── C. Landmark Localization (ResNet-18 Heatmap Regression: OD & Fovea)
       ├── D. Lesion Segmentation (IDRiD ResUNet V3: HE, Hard EX, Soft EX, MA)
       └── E. Vessel Segmentation (DRIVE Binary Vessel UNet)
                       │
                       ▼
      [ 4. Multi-Modal Evidence Fusion Engine ]
        - Synthesizes Attention + Contours + Landmarks + Vessels
        - Generates 5 distinct visual inspection layers
                       │
                       ▼
     [ 5. Tele-Screening Delivery & Triage Interface ]
       - Interactive Web Dashboard (Flask + SQLite)
       - Printable HTML Screening Report & 1080p Composite Summary PNG
       - Specialist Review Queue & Tele-Ophthalmology Workflow
```

---

## 3. Production vs. Research Mode Architecture

To guarantee strict backward compatibility and clinical safety, NetraSetu implements a dual-mode architectural separation:

| Dimension | Production Mode (`NETRASETU_RESEARCH_MODE=false`) | Research Mode (`NETRASETU_RESEARCH_MODE=true`) |
| :--- | :--- | :--- |
| **Intended Use** | Primary frontline clinical screening simulation | Research exploration & multi-modal explainability |
| **Classifier** | `NetraSetu_ResNet50_best.pth` (Locked Baseline) | Baseline + Offline Comparison Classifiers (V1, V2) |
| **Referable Threshold** | **`0.24` (Locked and Preserved)** | `0.24` (Locked and Preserved) |
| **Quality Gate** | Active (Instant REJECT fail-safe) | Active (Instant REJECT fail-safe) |
| **Explainability** | Grad-CAM + Primary Quality Diagnostics | Full Multi-Modal Studio (Grad-CAM, OD/Fovea, UNet V3, Vessels) |
| **Reports** | Standard screening summary & printable report | Multi-layer composite visual cards & HTML exports |
| **Audit Status** | Zero-mutation verified via SHA-256 digests | Fully validated via 6/6 regression test suite |

---

## 4. Datasets & Model Checkpoints

### A. Evaluated Datasets
1. **APTOS 2019 Blindness Detection:** 3,088 processed images (Train: 2,161, Validation: 463, Test: 464).
2. **IDRiD (Indian Diabetic Retinopathy Image Dataset):**
   - Disease Grading: 413 training images, 103 locked test images.
   - C. Localization: 413 development images, 103 locked test images with ground-truth OD/Fovea coordinates.
   - A. Segmentation: 54 development images (43 train / 11 val), 27 locked test images with pixel annotations.
3. **DRIVE (Digital Retinal Images for Vessel Extraction):** 40 images (20 train/val, 20 locked test images).

### B. Verified Model Checkpoints (`models/`)
- `NetraSetu_ResNet50_best.pth` (94.4 MB) — Production Baseline Classifier (**82.97% test accuracy**).
- `NetraSetu_ResNet50_final.pth` (94.4 MB) — Production Final Epoch Checkpoint.
- `NetraSetu_ResNet50_combined_best.pth` (282.7 MB) — Combined V1 Research Classifier (79.96% accuracy).
- `NetraSetu_ResNet50_combined_v2_best.pth` (282.7 MB) — Quality-Filtered V2 Classifier (76.72% accuracy).
- `NetraSetu_IDRiD_Localization_best.pth` (171.7 MB) — ResNet-18 Anatomical Landmark Regression.
- `NetraSetu_IDRiD_UNet_V3_best.pth` (159.5 MB) — ResUNet V3 4-Channel Lesion Segmentation (**0.2965 Macro Dice**).
- `NetraSetu_DRIVE_Vessel_UNet_best.pth` (89.0 MB) — DRIVE Binary Vessel Segmentation (**93.73% Accuracy**).

---

## 5. Experimental Results & Benchmark Metrics

> [!NOTE]
> All results represent **research evaluations on locked test sets and held-out validation splits**. Clinical validation has not been performed.

### A. DR Severity Classification & Referral Screening (APTOS Test Set)
- **5-Class Accuracy:** `82.97%`
- **Referable Sensitivity:** `96.64%` (Prioritizing rural screening sensitivity)
- **Referable Specificity:** `88.89%`
- **Referable F1-Score:** `87.80%`
- **Referable ROC-AUC:** `0.9778`
- **Referable Threshold:** `0.24`

### B. Post-Hoc Confidence Calibration (Temperature Scaling)
- **Optimal Temperature ($T$):** `0.9215` (Optimized via L-BFGS NLL minimization on 463 validation samples)
- **Expected Calibration Error (ECE):** Reduced from `0.0422` to **`0.0352`** (**-16.6% error reduction**)
- **Negative Log-Likelihood (NLL):** Reduced from `0.4541` to `0.4503`
- **Brier Multi-Class Score:** Reduced from `0.2256` to `0.2239`

### C. Anatomical Landmark Localization (IDRiD Test Set)
- **Optic Disc (OD) Mean Error:** `18.00 px` (Selected evaluation criterion: PCK@2% = 99.03%)
- **Fovea Centralis Mean Error:** `42.42 px` (Selected evaluation criterion: PCK@2% = 92.23%)
- **Combined Mean Error:** `30.21 px`

### D. Multi-Class Lesion Segmentation (IDRiD UNet V3 Test Set)
- **Hemorrhages (HE) Dice:** `0.3727`
- **Hard Exudates (EX) Dice:** `0.4660`
- **Soft Exudates (SE) Dice:** `0.3161`
- **Microaneurysms (MA) Dice:** `0.0311` (Transparently reported; sub-15px punctate lesions exhibit high pixel sparsity)
- **Macro Mean Dice:** **`0.2965`** (Outperforms UNet V2: `0.2831`, $\Delta = +0.0134$)

### E. Retinal Vessel Segmentation (DRIVE Validation Set)
- **Pixel Accuracy:** `93.73%`
- **Dice Coefficient:** `0.6780`
- **Intersection over Union (IoU):** `0.5132`
- **Specificity:** `95.79%`
- **ROC-AUC:** `0.9450`

---

## 6. Rural Capacity Planning & Triage Simulation

- **Single-Patient GPU Latency:** **325.2 ms** (Quality Gate: 5.2 ms, ResNet-50: 24.5 ms, Calibration: 0.4 ms, Grad-CAM: 42.1 ms, Landmarks: 19.8 ms, Lesions: 64.2 ms, Vessels: 21.0 ms, Fusion/Reports: 148.0 ms).
- **Continuous GPU Throughput:** **3.08 patients/sec** (~11,070 patients/hour).
- **100,000 Screenings/Year:** 50 patients/day over 2,000 active hours $\to$ requires **~0.45% nominal modeled compute utilization** on 1 edge workstation (RTX 3050).
- **250,000 Screenings/Year:** 125 patients/day $\to$ requires **~1.13% nominal modeled compute utilization**.
- **Engineering Disclosure:** *These figures are parametric engineering estimates based on configured timing assumptions and do not represent measured production deployment capacity.*
- **Illustrative Triage Funnel:** Under assumed rural epidemiological distributions (72.5% Grade 0, 9.8% Grade 1, 11.4% Grade 2, 4.1% Grade 3, 2.2% Grade 4), the model estimates that **82.3% of routine cases** can be managed at Primary Health Centers (PHCs), allowing specialist hospital visits to prioritize referable pathology.
- **Illustrative Economic Impact:** Assuming a 6.4% optical recapture rate, immediate alerts eliminate estimated return travel journeys, modeling **Rs. 2,880,000 INR** in avoided rural transit and lost daily wages per 100,000 screenings.

---

## 7. Installation & Quick Start

### Prerequisites
- Windows 10/11 (PowerShell) or Linux (Bash)
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ / 12.0+ (Optional; CPU fallback supported)

### Step 1: Clone and Set Up Virtual Environment
```powershell
cd .
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Run Production Web Application
```powershell
# Default mode (NETRASETU_RESEARCH_MODE=false)
python app.py
```
Open browser at: `http://127.0.0.1:5000/`

### Step 3: Run with Research Mode Enabled
```powershell
$env:NETRASETU_RESEARCH_MODE="true"
python app.py
```

### Step 4: Run CLI Research Pipeline
```powershell
# Process single fundus image and generate HTML/PNG reports
python src\research_pipeline.py --image data\APTOS\raw\train_images\002c21358ce6.png --output_dir results\screening\
```

---

## 8. Directory Layout

```
.\
├── app.py                                         # Flask web application & API endpoints
├── models\                                        # Model checkpoints (production + research)
│   ├── NetraSetu_ResNet50_best.pth                # Baseline production classifier (LOCKED)
│   ├── NetraSetu_IDRiD_Localization_best.pth      # ResNet-18 OD & Fovea localization
│   ├── NetraSetu_IDRiD_UNet_V3_best.pth           # ResUNet V3 lesion segmentation
│   └── NetraSetu_DRIVE_Vessel_UNet_best.pth       # DRIVE binary vessel segmentation
├── src\                                           # Core source code modules
│   ├── inference.py                               # Core baseline preprocessing & classification
│   ├── inference_service.py                       # Production singleton service & caching
│   ├── explainability_engine.py                   # Unified 9-component multi-modal engine
│   ├── research_pipeline.py                       # End-to-end research screening CLI
│   ├── generate_explainable_report.py             # Printable HTML & 1080p composite generator
│   ├── capacity_simulator.py                      # 100k+ rural capacity & triage simulator
│   └── calibrate_resnet50.py                      # Temperature scaling optimizer
├── results\final_pipeline\                        # Official deliverables & evaluation outputs
│   ├── calibration\                               # Temperature scaling curves & metrics
│   ├── segmentation\                              # UNet V3 test results & threshold JSON
│   ├── vessel_segmentation\                       # DRIVE test metrics & 12 overlay examples
│   ├── explainability\                            # 60 exported visual evidence layers
│   ├── capacity\                                  # Capacity model, text report, 4-panel plot
│   ├── qa\                                        # Audit reports & 6/6 regression test logs
│   └── reports\                                   # Coverage matrix & executive summaries
├── templates\ & static\                           # Modern responsive UI assets
└── backup\                                        # Archived baseline snapshots
```

---

## 9. Known Limitations & Research Governance

1. **Clinical Validation Status:** NetraSetu has **not undergone clinical validation or human clinical trials**. It is not cleared or approved by any medical regulatory agency (CDSCO, FDA, or CE) as a medical diagnostic device.
2. **Domain Specificity:** The OD/Fovea localization model and UNet V3 lesion segmentation models were trained on IDRiD images. While they produce indicative features on other datasets, cross-domain inferences must be interpreted as research prototypes.
3. **Microaneurysm Detection:** Microaneurysms (< 15 pixels) exhibit high spatial sparsity; while larger hemorrhages and exudates show strong delineation, microaneurysm pixel-level Dice remains low (0.0311).
4. **Physician Supervision Mandatory:** All outputs, predictions, risk scores, and visual overlays must be reviewed and confirmed by a certified ophthalmologist before any medical intervention.

---

## 10. Team & SIH Acknowledgements

- **Event:** Smart India Hackathon (SIH) 2026
- **Problem Statement ID:** 26038
- **Project Title:** NetraSetu — Explainable AI for Diabetic Retinopathy Screening in Rural India
