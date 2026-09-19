# NetraSetu Datasets Directory

> **Notice:** Datasets are not included in this repository.

Due to file size and dataset licensing terms, fundus images and annotation archives are excluded from version control via `.gitignore`.

---

## Dataset Sources & Expected Layout

To run local training or evaluations, arrange datasets into the following folder structure under `data/`:

```text
data/
├── APTOS/
│   ├── raw/
│   │   ├── train_images/               # 3,662 original fundus images (.png)
│   │   └── train.csv                   # Image ID and DR grade (0-4)
│   └── processed/
│       ├── train/                      # 2,161 quality-screened training images
│       ├── val/                        # 463 quality-screened validation images
│       └── test/                       # 464 quality-screened test images
│
├── IDRiD_Grading/
│   └── B. Disease Grading/
│       ├── 1. Original Images/
│       │   ├── a. Training Set/        # 413 fundus photographs (.jpg)
│       │   └── b. Testing Set/         # 103 fundus photographs (.jpg)
│       └── 2. Groundtruths/
│           ├── a. IDRiD_Disease Grading_Training Labels.csv
│           └── b. IDRiD_Disease Grading_Testing Labels.csv
│
├── IDRiD_Localization/
│   ├── train/                          # 351 development images
│   ├── val/                            # 62 development images
│   └── test/                           # 103 locked test images with OD/Fovea coordinates
│
├── IDRiD/ (Segmentation)
│   ├── A. Segmentation/
│   │   ├── 1. Original Images/
│   │   │   ├── a. Training Set/        # 54 training fundus images
│   │   │   └── b. Testing Set/         # 27 testing fundus images
│   │   └── 2. All Lesion Groundtruths/ # Binary masks for MA, HE, EX, SE
│
└── DRIVE/
    ├── train/                          # 17 training images + manual vessel groundtruth
    ├── val/                            # 3 validation images
    └── test/                           # 20 test images + 1st/2nd manual vessel annotations
```

---

## Benchmark Citations & Attributions

1. **APTOS 2019 Blindness Detection:**
   - Asia Pacific Tele-Ophthalmology Society (Kaggle Competition Dataset).
2. **IDRiD (Indian Diabetic Retinopathy Image Dataset):**
   - Porwal et al., *"Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research"*, Data 2018, 3(3), 25.
3. **DRIVE (Digital Retinal Images for Vessel Extraction):**
   - Staal et al., *"Ridge based vessel detection in color images of the retina"*, IEEE Trans Med Imaging, 2004.
