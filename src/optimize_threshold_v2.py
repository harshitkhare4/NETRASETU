import os
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix


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
print("NETRASETU THRESHOLD OPTIMIZATION V2")
print("============================================")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


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

dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Validation images:", len(dataset))


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# PREDICTIONS
# ============================================================

labels = []
scores = []

with torch.no_grad():

    for images, y in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Referable probability
        # P(Level 2) + P(Level 3) + P(Level 4)

        referable_score = (
            probabilities[:, 2]
            + probabilities[:, 3]
            + probabilities[:, 4]
        )

        labels.extend(
            y.numpy()
        )

        scores.extend(
            referable_score.cpu().numpy()
        )


y_true = np.array(labels)

scores = np.array(scores)

y_true = (
    y_true >= 2
).astype(int)


# ============================================================
# SEARCH ALL THRESHOLDS
# ============================================================

results = []

for threshold in np.arange(
    0.05,
    0.96,
    0.01
):

    y_pred = (
        scores >= threshold
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

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2

    youden = (
        sensitivity + specificity - 1
    )

    results.append([
        threshold,
        sensitivity,
        specificity,
        balanced_accuracy,
        youden,
        tp,
        fp,
        fn,
        tn
    ])


results = np.array(results)


# ============================================================
# OPTION 1:
# Highest sensitivity while validation specificity >= 90%
# ============================================================

eligible = results[
    results[:, 2] >= 0.90
]

print("\n============================================")
print("TARGET-ORIENTED SEARCH")
print("============================================")

if len(eligible) > 0:

    index = np.argmax(
        eligible[:, 1]
    )

    best = eligible[index]

    threshold = best[0]
    sensitivity = best[1]
    specificity = best[2]

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Validation Sensitivity: "
        f"{sensitivity * 100:.2f}%"
    )

    print(
        f"Validation Specificity: "
        f"{specificity * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{best[3] * 100:.2f}%"
    )

    print(
        f"Youden J: "
        f"{best[4]:.4f}"
    )

else:

    print(
        "\nNo threshold gives validation "
        "specificity >= 90%."
    )


# ============================================================
# OPTION 2:
# BEST BALANCED ACCURACY
# ============================================================

index = np.argmax(
    results[:, 3]
)

balanced_best = results[index]

print("\n============================================")
print("BEST BALANCED-ACCURACY THRESHOLD")
print("============================================")

print(
    f"Threshold: "
    f"{balanced_best[0]:.2f}"
)

print(
    f"Sensitivity: "
    f"{balanced_best[1] * 100:.2f}%"
)

print(
    f"Specificity: "
    f"{balanced_best[2] * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_best[3] * 100:.2f}%"
)


# ============================================================
# SHOW CANDIDATE THRESHOLDS
# ============================================================

print("\n============================================")
print("USEFUL THRESHOLD OPTIONS")
print("============================================")

print(
    "\nThreshold | Sensitivity | Specificity"
)

print(
    "----------------------------------------"
)

for row in results:

    threshold = row[0]

    # Show thresholds from 0.10 to 0.50
    # at steps of 0.05 approximately

    if (
        threshold >= 0.10
        and threshold <= 0.50
        and abs(
            (threshold * 100) % 5
        ) < 0.01
    ):

        print(
            f"{threshold:8.2f} | "
            f"{row[1] * 100:10.2f}% | "
            f"{row[2] * 100:9.2f}%"
        )


# ============================================================
# SAVE TARGET-ORIENTED THRESHOLD
# ============================================================

if len(eligible) > 0:

    selected_threshold = float(
        eligible[
            np.argmax(eligible[:, 1])
        ][0]
    )

else:

    # Fall back to best balanced accuracy

    selected_threshold = float(
        balanced_best[0]
    )


threshold_path = os.path.join(
    RESULT_DIR,
    "referable_threshold_v2.txt"
)

with open(
    threshold_path,
    "w"
) as f:

    f.write(
        str(selected_threshold)
    )


print("\n============================================")
print("SELECTED THRESHOLD")
print("============================================")

print(
    f"Threshold: {selected_threshold:.2f}"
)

print(
    "\nSaved to:"
)

print(
    threshold_path
)

print(
    "============================================"
)