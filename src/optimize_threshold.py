import os
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix


# ============================================================
# NETRASETU - REFERABLE DR THRESHOLD OPTIMIZATION
#
# IMPORTANT:
# Threshold is selected using VALIDATION data only.
# Test data is NOT used for threshold selection.
# ============================================================


PROJECT_ROOT = r"C:\NetraSetu"

VAL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "APTOS",
    "processed",
    "val"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "NetraSetu_ResNet50_best.pth"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n============================================")
print("NETRASETU THRESHOLD OPTIMIZATION")
print("============================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# VALIDATION DATASET
# ============================================================

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("\nValidation images:", len(val_dataset))

print(
    "Classes:",
    val_dataset.classes
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    5
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Model loaded.")


# ============================================================
# GET VALIDATION PROBABILITIES
# ============================================================

all_labels = []
all_referable_scores = []


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Referable probability:
        # Level 2 + Level 3 + Level 4

        referable_score = (
            probabilities[:, 2]
            + probabilities[:, 3]
            + probabilities[:, 4]
        )

        all_referable_scores.extend(
            referable_score.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


y_true_multiclass = np.array(
    all_labels
)

referable_scores = np.array(
    all_referable_scores
)


# ============================================================
# TRUE BINARY LABELS
# ============================================================

y_true = (
    y_true_multiclass >= 2
).astype(int)


# ============================================================
# SEARCH THRESHOLDS
# ============================================================

best_threshold = None

best_sensitivity = -1

best_specificity = -1


print(
    "\nSearching thresholds..."
)

for threshold in np.arange(
    0.05,
    0.96,
    0.01
):

    y_pred = (
        referable_scores >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # Requirement:
    # Specificity must remain at least 85%.
    #
    # Among those thresholds, choose the one
    # giving the highest sensitivity.

    if specificity >= 0.85:

        if sensitivity > best_sensitivity:

            best_sensitivity = sensitivity

            best_specificity = specificity

            best_threshold = threshold


# ============================================================
# RESULT
# ============================================================

print(
    "\n============================================"
)

if best_threshold is None:

    print(
        "No threshold found with specificity >= 85%."
    )

else:

    print(
        f"Best threshold: {best_threshold:.2f}"
    )

    print(
        f"Validation sensitivity: "
        f"{best_sensitivity * 100:.2f}%"
    )

    print(
        f"Validation specificity: "
        f"{best_specificity * 100:.2f}%"
    )


# ============================================================
# SAVE THRESHOLD
# ============================================================

if best_threshold is not None:

    threshold_file = os.path.join(
        RESULT_DIR,
        "referable_threshold.txt"
    )

    with open(
        threshold_file,
        "w"
    ) as f:

        f.write(
            str(best_threshold)
        )

    print(
        "\nThreshold saved to:"
    )

    print(
        threshold_file
    )


print(
    "\n============================================"
)
print(
    "THRESHOLD OPTIMIZATION COMPLETE"
)
print(
    "============================================"
)