import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet50_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# NETRASETU - DIABETIC RETINOPATHY CLASSIFICATION
# ResNet-50 + PyTorch + RTX 3050
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\NetraSetu"

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "APTOS",
    "processed",
    "train"
)

VAL_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "APTOS",
    "processed",
    "val"
)

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "APTOS",
    "processed",
    "test"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. DEVICE
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n============================================")
print("NETRASETU DR TRAINING")
print("============================================")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )


# ------------------------------------------------------------
# 3. IMAGE SIZE
# ------------------------------------------------------------

IMAGE_SIZE = 224


# ------------------------------------------------------------
# 4. TRANSFORMS
# ------------------------------------------------------------

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),

    A.Rotate(
        limit=10,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    ),

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()
])


val_transform = A.Compose([
    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()
])


# ------------------------------------------------------------
# 5. CUSTOM DATASET
# ------------------------------------------------------------

class AlbumentationsDataset(datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albumentations_transform = transform

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = self.loader(image_path)

        image = np.array(image)

        if self.albumentations_transform:

            transformed = self.albumentations_transform(
                image=image
            )

            image = transformed["image"]

        return image, label


# ------------------------------------------------------------
# 6. DATASETS
# ------------------------------------------------------------

train_dataset = AlbumentationsDataset(
    TRAIN_DIR,
    train_transform
)

val_dataset = AlbumentationsDataset(
    VAL_DIR,
    val_transform
)

test_dataset = AlbumentationsDataset(
    TEST_DIR,
    val_transform
)


print("\nClasses:")
print(train_dataset.classes)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# ------------------------------------------------------------
# 7. DATALOADERS
# ------------------------------------------------------------

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ------------------------------------------------------------
# 8. CLASS WEIGHTS
# ------------------------------------------------------------

train_labels = np.array(
    [label for _, label in train_dataset.samples]
)

class_counts = np.bincount(
    train_labels,
    minlength=5
)

print("\nClass counts:")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count}")


# Inverse-frequency weighting
class_weights = (
    len(train_labels)
    /
    (len(class_counts) * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass weights:")
print(class_weights)


# ------------------------------------------------------------
# 9. LOAD PRETRAINED RESNET-50
# ------------------------------------------------------------

print("\nLoading ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)


# ------------------------------------------------------------
# 10. REPLACE FINAL CLASSIFIER
# ------------------------------------------------------------

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    5
)

model = model.to(device)


# ------------------------------------------------------------
# 11. LOSS FUNCTION
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ------------------------------------------------------------
# 12. OPTIMIZER
# ------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# ------------------------------------------------------------
# 13. LEARNING RATE SCHEDULER
# ------------------------------------------------------------

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ------------------------------------------------------------
# 14. TRAINING SETTINGS
# ------------------------------------------------------------

EPOCHS = 10

best_val_loss = float("inf")

best_model_weights = copy.deepcopy(
    model.state_dict()
)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}


# ------------------------------------------------------------
# 15. TRAINING LOOP
# ------------------------------------------------------------

for epoch in range(EPOCHS):

    print(
        f"\n\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 50)

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    train_progress = tqdm(
        train_loader,
        desc="Training"
    )

    for images, labels in train_progress:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        train_progress.set_postfix(
            loss=loss.item()
        )

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        val_progress = tqdm(
            val_loader,
            desc="Validation"
        )

        for images, labels in val_progress:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item()
                *
                images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(val_loss)


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_acc"].append(
        train_accuracy
    )

    history["val_acc"].append(
        val_accuracy
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nTrain Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Save BEST model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "class_to_idx":
                    train_dataset.class_to_idx,

                "image_size":
                    IMAGE_SIZE
            },
            os.path.join(
                MODEL_DIR,
                "NetraSetu_ResNet50_best.pth"
            )
        )

        print(
            "✅ Best model saved."
        )


# ------------------------------------------------------------
# 16. LOAD BEST MODEL
# ------------------------------------------------------------

model.load_state_dict(
    best_model_weights
)


# ------------------------------------------------------------
# 17. SAVE FINAL MODEL
# ------------------------------------------------------------

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "classes":
            train_dataset.classes,

        "class_to_idx":
            train_dataset.class_to_idx,

        "image_size":
            IMAGE_SIZE
    },
    os.path.join(
        MODEL_DIR,
        "NetraSetu_ResNet50_final.pth"
    )
)


# ------------------------------------------------------------
# 18. TRAINING GRAPHS
# ------------------------------------------------------------

epochs_range = range(
    1,
    EPOCHS + 1
)


# Loss graph

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "NetraSetu Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "loss_curve.png"
    ),
    dpi=300
)

plt.close()


# Accuracy graph

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "NetraSetu Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy_curve.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 19. FINISHED
# ------------------------------------------------------------

print("\n============================================")
print("TRAINING COMPLETED")
print("============================================")

print(
    "\nBest model:"
)

print(
    os.path.join(
        MODEL_DIR,
        "NetraSetu_ResNet50_best.pth"
    )
)

print(
    "\nFinal model:"
)

print(
    os.path.join(
        MODEL_DIR,
        "NetraSetu_ResNet50_final.pth"
    )
)

print(
    "\nResults:"
)

print(
    os.path.join(
        RESULT_DIR,
        "loss_curve.png"
    )
)

print(
    os.path.join(
        RESULT_DIR,
        "accuracy_curve.png"
    )
)