import os
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score
)


# ============================================================
# NETRASETU
# FINAL TEST EVALUATION USING VALIDATION-SELECTED THRESHOLD
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

THRESHOLD_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "referable_threshold.txt"
)


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n============================================")
print("NETRASETU FINAL TEST EVALUATION")
print("============================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 2. READ THRESHOLD
# ============================================================

with open(
    THRESHOLD_FILE,
    "r"
) as f:
    threshold = float(f.read().strip())

print(
    f"Using validation-selected threshold: {threshold:.2f}"
)


# ============================================================
# 3. TEST TRANSFORM
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
# 4. TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("\nTest images:", len(test_dataset))

print(
    "Classes:",
    test_dataset.classes
)


# ============================================================
# 5. LOAD MODEL
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

print("Model loaded successfully.")


# ============================================================
# 6. PREDICTION
# ============================================================

all_labels = []
all_predictions_5class = []
all_referable_scores = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        # Normal 5-class prediction
        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # ----------------------------------------------------
        # REFERABLE SCORE
        #
        # Level 2 + Level 3 + Level 4
        # ----------------------------------------------------

        referable_score = (
            probabilities[:, 2]
            + probabilities[:, 3]
            + probabilities[:, 4]
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions_5class.extend(
            predictions.cpu().numpy()
        )

        all_referable_scores.extend(
            referable_score.cpu().numpy()
        )


y_true = np.array(
    all_labels
)

y_pred_5class = np.array(
    all_predictions_5class
)

referable_scores = np.array(
    all_referable_scores
)


# ============================================================
# 7. NORMAL 5-CLASS RESULTS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred_5class
)

print("\n============================================")
print("5-CLASS RESULTS")
print("============================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred_5class,
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
# 8. REFERABLE DR
#
# Actual:
# Level 0,1 = 0
# Level 2,3,4 = 1
#
# Prediction:
# score >= threshold = 1
# ============================================================

y_true_binary = (
    y_true >= 2
).astype(int)

y_pred_binary = (
    referable_scores >= threshold
).astype(int)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true_binary,
    y_pred_binary,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# 10. METRICS
# ============================================================

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

precision = precision_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

f1 = f1_score(
    y_true_binary,
    y_pred_binary,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true_binary,
    referable_scores
)


# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print("\n============================================")
print("FINAL REFERABLE DR RESULTS")
print("============================================")

print(
    "Level 0-1 = Non-Referable"
)

print(
    "Level 2-4 = Referable"
)

print(
    f"\nThreshold: {threshold:.2f}"
)

print(
    f"Sensitivity: {sensitivity * 100:.2f}%"
)

print(
    f"Specificity: {specificity * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"F1 Score: {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nBinary Confusion Matrix:")

print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# 13. SIH TARGET CHECK
# ============================================================

print("\n============================================")
print("SIH TARGET CHECK")
print("============================================")

if sensitivity > 0.90:
    print(
        "Sensitivity >90%: PASSED"
    )
else:
    print(
        "Sensitivity >90%: NOT REACHED"
    )


if specificity > 0.85:
    print(
        "Specificity >85%: PASSED"
    )
else:
    print(
        "Specificity >85%: NOT REACHED"
    )


# ============================================================
# 14. SAVE FINAL RESULTS
# ============================================================

result_path = os.path.join(
    PROJECT_ROOT,
    "results",
    "final_test_threshold_results.txt"
)

with open(
    result_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NETRASETU FINAL TEST RESULTS\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Threshold: {threshold:.4f}\n"
    )

    f.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Sensitivity: {sensitivity * 100:.2f}%\n"
    )

    f.write(
        f"Specificity: {specificity * 100:.2f}%\n"
    )

    f.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score: {f1 * 100:.2f}%\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


print(
    "\nResults saved to:"
)

print(result_path)

print(
    "\n============================================"
)
print(
    "FINAL TEST EVALUATION COMPLETE"
)
print(
    "============================================"
)