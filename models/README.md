# NetraSetu Model Checkpoints

> **Notice:** Model weights are not included in this repository.

Due to file size constraints, storage quotas, and repository management considerations, pre-trained PyTorch weight checkpoints (`*.pth`) are intentionally excluded from version control via `.gitignore`.

---

## Expected Local Checkpoints

When running NetraSetu locally, place trained weights in this `models/` directory:

### 1. Production Classifier (Baseline)
- `NetraSetu_ResNet50_best.pth` *(Primary Production Model, 90.02 MB, ResNet-50 trained on quality-screened APTOS 2019)*
- `NetraSetu_ResNet50_final.pth` *(Final epoch training checkpoint)*

### 2. Experimental Disease Grading Models
- `NetraSetu_ResNet50_combined_best.pth` *(Combined V1: APTOS + unfiltered IDRiD)*
- `NetraSetu_ResNet50_combined_final.pth` *(Combined V1 final checkpoint)*
- `NetraSetu_ResNet50_combined_v2_best.pth` *(Combined V2: APTOS + quality-screened IDRiD)*
- `NetraSetu_ResNet50_combined_v2_final.pth` *(Combined V2 final checkpoint)*

### 3. Anatomical Landmark Localization
- `NetraSetu_IDRiD_Localization_best.pth` *(ResNet-18 dual regression for Optic Disc & Fovea coordinates)*
- `NetraSetu_IDRiD_Localization_final.pth` *(Localization final checkpoint)*

### 4. Multi-Class Retinal Lesion Segmentation
- `NetraSetu_IDRiD_UNet_best.pth` *(UNet V1 baseline)*
- `NetraSetu_IDRiD_UNet_V2_best.pth` *(UNet V2)*
- `NetraSetu_IDRiD_UNet_V3_best.pth` *(UNet V3 multi-class: Microaneurysms, Hemorrhages, Hard Exudates, Soft Exudates)*
- `NetraSetu_IDRiD_UNet_V3_final.pth` *(UNet V3 final checkpoint)*

### 5. Retinal Vessel Segmentation
- `NetraSetu_DRIVE_Vessel_UNet_best.pth` *(Modified UNet with CLAHE enhancement trained on DRIVE)*
- `NetraSetu_DRIVE_Vessel_UNet_final.pth` *(Vessel segmentation final checkpoint)*

---

## Architecture Specifications

| Module | Model Architecture | Input Resolution | Output Dimension | Checkpoint Size |
| :--- | :--- | :--- | :--- | :--- |
| **DR Classifier** | Deep ResNet-50 (Pretrained transfer backbone) | 512 × 512 × 3 | 5 Severity Classes (0–4) | ~90 MB |
| **Localization** | ResNet-18 Landmark Regressor | 512 × 512 × 3 | 4 Coordinates ($x_{od}, y_{od}, x_{fov}, y_{fov}$) | ~43 MB |
| **Lesion UNet** | UNet (4-channel sigmoid multi-label) | 512 × 512 × 3 | 4 Masks ($512 \times 512$) | ~118 MB |
| **Vessel UNet** | Single-channel UNet (Binary vessel mask) | 512 × 512 × 1 | 1 Binary Mask ($512 \times 512$) | ~54 MB |

For reproducible training scripts, consult the [`src/`](../src/) directory.
