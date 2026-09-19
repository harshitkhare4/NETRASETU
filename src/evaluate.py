import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt


# ============================================================
# NETRASETU - MODEL EVALUATION
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "APTOS",
    "processed",
    "test"
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
print("NETRASETU MODEL EVALUATION")
print("============================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# TRANSFORM
# Same basic preprocessing expected by the model
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("\nClasses:")
print(test_dataset.classes)

print("\nClass mapping:")
print(test_dataset.class_to_idx)

print("\nTest images:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet-50...")

model = models.resnet50(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
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

print("Model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []
all_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


y_true = np.array(all_labels)

y_pred = np.array(all_predictions)

y_prob = np.array(all_probabilities)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n============================================")
print("OVERALL RESULTS")
print("============================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=[
            "Level 0",
            "Level 1",
            "Level 2",
            "Level 3",
            "Level 4"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")

print(cm)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "NetraSetu - DR Confusion Matrix"
)

plt.colorbar()

class_names = [
    "Level 0",
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4"
]

plt.xticks(
    range(5),
    class_names
)

plt.yticks(
    range(5),
    class_names
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

for i in range(5):

    for j in range(5):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# REFERABLE DR
#
# Level 0,1 -> Non-Referable
# Level 2,3,4 -> Referable
# ============================================================

y_true_binary = (
    y_true >= 2
).astype(int)

y_pred_binary = (
    y_pred >= 2
).astype(int)


# Confusion matrix
binary_cm = confusion_matrix(
    y_true_binary,
    y_pred_binary,
    labels=[0, 1]
)

tn, fp, fn, tp = binary_cm.ravel()


# ============================================================
# SENSITIVITY
# ============================================================

sensitivity = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)


# ============================================================
# SPECIFICITY
# ============================================================

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# ============================================================
# BINARY PRECISION / F1
# ============================================================

precision = precision_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

recall = recall_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

f1 = f1_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n============================================")
print("REFERABLE DR RESULTS")
print("============================================")

print("Level 0-1 = Non-Referable")
print("Level 2-4 = Referable")

print("\nBinary Confusion Matrix:")
print(binary_cm)

print(
    f"\nSensitivity: {sensitivity * 100:.2f}%"
)

print(
    f"Specificity: {specificity * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall: {recall * 100:.2f}%"
)

print(
    f"F1 Score: {f1 * 100:.2f}%"
)


# ============================================================
# CHECK TARGETS
# ============================================================

print("\n============================================")
print("SIH TARGET CHECK")
print("============================================")

if sensitivity >= 0.90:
    print("Sensitivity target >90%: PASSED")
else:
    print("Sensitivity target >90%: NOT YET REACHED")

if specificity >= 0.85:
    print("Specificity target >85%: PASSED")
else:
    print("Specificity target >85%: NOT YET REACHED")


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

result_file = os.path.join(
    RESULT_DIR,
    "evaluation_results.txt"
)

with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NETRASETU MODEL EVALUATION\n"
    )

    f.write(
        "===========================\n\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Sensitivity: "
        f"{sensitivity * 100:.2f}%\n"
    )

    f.write(
        f"Specificity: "
        f"{specificity * 100:.2f}%\n"
    )

    f.write(
        f"Precision: "
        f"{precision * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score: "
        f"{f1 * 100:.2f}%\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nReferable DR Confusion Matrix:\n"
    )

    f.write(
        str(binary_cm)
    )


print(
    "\nResults saved to:"
)

print(result_file)

print(
    "\nConfusion matrix saved to:"
)

print(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    )
)

print(
    "\n============================================"
)

print(
    "EVALUATION COMPLETE"
)

print(
    "============================================"
)