import os
import copy
import json
import time
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
# NETRASETU - COMBINED RESNET-50 DR CLASSIFICATION (APTOS + IDRiD)
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "combined_grading")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "combined_training")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "NetraSetu_ResNet50_combined_best.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "NetraSetu_ResNet50_combined_final.pth")

CONFIG_PATH = os.path.join(RESULTS_DIR, "training_config.json")
RESULTS_TXT_PATH = os.path.join(RESULTS_DIR, "training_results.txt")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss_curve.png")
ACC_CURVE_PATH = os.path.join(RESULTS_DIR, "accuracy_curve.png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. HARDWARE & DEVICE
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n==================================================")
print("NETRASETU COMBINED RESNET-50 TRAINING (APTOS + IDRiD)")
print("==================================================")
print(f"PyTorch Version: {torch.__version__}")
print(f"Device:          {device}")

if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    total_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    print(f"GPU Total VRAM:  {total_mem} GB")
    # Modern PyTorch AMP support
    use_amp = True
    scaler = torch.amp.GradScaler('cuda')
else:
    use_amp = False
    scaler = None

# ------------------------------------------------------------
# 2. HYPERPARAMETERS & CONFIGURATION
# ------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 12
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ------------------------------------------------------------
# 3. ALBUMENTATIONS TRANSFORMS
# ------------------------------------------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ------------------------------------------------------------
# 4. CUSTOM DATASET CLASS
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
            transformed = self.albumentations_transform(image=image)
            image = transformed["image"]

        return image, label

# ------------------------------------------------------------
# 5. DATA LOADERS & DATASET SIZES
# ------------------------------------------------------------
train_dataset = AlbumentationsDataset(TRAIN_DIR, train_transform)
val_dataset = AlbumentationsDataset(VAL_DIR, val_transform)

print(f"\nDataset Sizes:")
print(f"  Training Split:   {len(train_dataset)}")
print(f"  Validation Split: {len(val_dataset)}")
print(f"  Total Dev Pool:   {len(train_dataset) + len(val_dataset)}")
print(f"  Classes:          {train_dataset.classes}")
print(f"  Class to index:   {train_dataset.class_to_idx}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

# ------------------------------------------------------------
# 6. CLASS WEIGHTS (TRAINING SPLIT ONLY)
# ------------------------------------------------------------
train_labels = np.array([label for _, label in train_dataset.samples])
class_counts = np.bincount(train_labels, minlength=5)

print("\nTraining Split Class Counts:")
for i, count in enumerate(class_counts):
    print(f"  Class {i}: {count} ({round(count/len(train_labels)*100, 2)}%)")

# Inverse-frequency loss weighting: N_total / (5 * N_c)
class_weights = len(train_labels) / (5.0 * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"\nCalculated Class Weights (inverse frequency):")
for i, w in enumerate(class_weights):
    print(f"  Class {i}: {round(float(w), 4)}")

# ------------------------------------------------------------
# 7. MODEL DEFINITION & MODIFICATION
# ------------------------------------------------------------
print("\nInstantiating Pretrained ResNet-50...")
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ------------------------------------------------------------
# 8. TRAINING & VALIDATION LOOP
# ------------------------------------------------------------
best_val_loss = float("inf")
best_epoch = 0
best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "lr": []
}

start_training_time = time.time()
print(f"\nStarting {EPOCHS} Training Epochs (Batch Size: {BATCH_SIZE}, AMP: {use_amp})...")

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    current_lr = optimizer.param_groups[0]["lr"]

    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d} [Train]")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_train_loss = running_loss / total_samples
    epoch_train_acc = running_corrects / total_samples

    # --- Validation Phase ---
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data).item()
            val_total += inputs.size(0)

    epoch_val_loss = val_running_loss / val_total
    epoch_val_acc = val_running_corrects / val_total

    scheduler.step(epoch_val_loss)

    epoch_duration = time.time() - epoch_start_time

    history["train_loss"].append(round(epoch_train_loss, 4))
    history["val_loss"].append(round(epoch_val_loss, 4))
    history["train_acc"].append(round(epoch_train_acc * 100, 2))
    history["val_acc"].append(round(epoch_val_acc * 100, 2))
    history["lr"].append(current_lr)

    print(
        f"Epoch {epoch:02d}/{EPOCHS:02d} | "
        f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_duration:.1f}s"
    )

    # Checkpoint on best validation loss
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch = epoch
        best_val_acc = epoch_val_acc
        best_model_weights = copy.deepcopy(model.state_dict())

        best_checkpoint = {
            "epoch": best_epoch,
            "model_state_dict": best_model_weights,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "class_weights": class_weights.tolist(),
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "model_name": "ResNet-50 Combined (APTOS + IDRiD)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(best_checkpoint, BEST_MODEL_PATH)
        print(f"  [*] Saved new BEST combined model checkpoint to: {BEST_MODEL_PATH}")

total_training_duration = round(time.time() - start_training_time, 1)
print(f"\nTraining completed in {total_training_duration}s ({round(total_training_duration/60, 1)} min).")
print(f"Best Validation Loss: {best_val_loss:.4f} (Accuracy: {best_val_acc*100:.2f}%) at Epoch {best_epoch}")

# Save final checkpoint
final_checkpoint = {
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "final_val_loss": epoch_val_loss,
    "final_val_acc": epoch_val_acc,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "best_val_acc": best_val_acc,
    "class_weights": class_weights.tolist(),
    "classes": train_dataset.classes,
    "class_to_idx": train_dataset.class_to_idx,
    "model_name": "ResNet-50 Combined Final (APTOS + IDRiD)",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
torch.save(final_checkpoint, FINAL_MODEL_PATH)
print(f"Saved FINAL combined model checkpoint to: {FINAL_MODEL_PATH}")

# ------------------------------------------------------------
# 9. PLOT TRAINING CURVES
# ------------------------------------------------------------
epochs_range = range(1, EPOCHS + 1)

# Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
plt.plot(epochs_range, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
plt.title("NetraSetu ResNet-50 Combined: Training vs Validation Loss", fontsize=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("CrossEntropy Loss", fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(LOSS_CURVE_PATH, dpi=200)
plt.close()
print(f"Saved loss curve to: {LOSS_CURVE_PATH}")

# Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], "b-o", label="Training Accuracy (%)", linewidth=2)
plt.plot(epochs_range, history["val_acc"], "g--s", label="Validation Accuracy (%)", linewidth=2)
plt.title("NetraSetu ResNet-50 Combined: Training vs Validation Accuracy", fontsize=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ACC_CURVE_PATH, dpi=200)
plt.close()
print(f"Saved accuracy curve to: {ACC_CURVE_PATH}")

# ------------------------------------------------------------
# 10. SAVE CONFIGURATION & RESULTS SUMMARY
# ------------------------------------------------------------
config_data = {
    "model_architecture": "ResNet-50",
    "dataset": "APTOS (2161) + IDRiD (413) = 2574 total",
    "train_split": len(train_dataset),
    "val_split": len(val_dataset),
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": EPOCHS,
    "device": str(device),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    "mixed_precision": use_amp,
    "class_counts": class_counts.tolist(),
    "class_weights": class_weights.tolist(),
    "best_epoch": best_epoch,
    "best_val_loss": round(best_val_loss, 4),
    "best_val_accuracy_percent": round(best_val_acc * 100, 2),
    "total_training_duration_seconds": total_training_duration,
    "best_checkpoint_path": BEST_MODEL_PATH,
    "final_checkpoint_path": FINAL_MODEL_PATH
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=4)
print(f"Saved training configuration to: {CONFIG_PATH}")

results_text = f"""NETRASETU - RESNET-50 COMBINED TRAINING REPORT
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

1. SYSTEM & DEVICE:
- Device:                 {device}
- GPU Model:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
- CUDA Memory:            {total_mem if torch.cuda.is_available() else 0} GB
- Mixed Precision (AMP):  {use_amp}

2. DATASET CONFIGURATION:
- Training Split Size:    {len(train_dataset)} images (85.0%)
- Validation Split Size:  {len(val_dataset)} images (15.0%)
- Total Combined Pool:    {len(train_dataset) + len(val_dataset)} images
- Image Input Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}
- Batch Size:             {BATCH_SIZE}

3. CLASS DISTRIBUTION (TRAIN SPLIT):
- Class 0 (No DR):            {class_counts[0]} (Weight: {class_weights[0]:.4f})
- Class 1 (Mild DR):          {class_counts[1]} (Weight: {class_weights[1]:.4f})
- Class 2 (Moderate DR):      {class_counts[2]} (Weight: {class_weights[2]:.4f})
- Class 3 (Severe DR):        {class_counts[3]} (Weight: {class_weights[3]:.4f})
- Class 4 (Proliferative DR): {class_counts[4]} (Weight: {class_weights[4]:.4f})

4. TRAINING HYPERPARAMETERS:
- Optimizer:              AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})
- Loss Function:          Weighted CrossEntropyLoss
- Learning Rate Schedule: ReduceLROnPlateau(mode='min', factor=0.5, patience=2)
- Augmentations:          HorizontalFlip(0.5), Rotate(±10°), RandomBrightnessContrast(±15%)
- Total Epochs:           {EPOCHS}

5. TRAINING HISTORY (EPOCH-BY-EPOCH):
"""

for ep in range(EPOCHS):
    results_text += (
        f"Epoch {ep+1:02d}: Train Loss = {history['train_loss'][ep]:.4f}, "
        f"Train Acc = {history['train_acc'][ep]:.2f}%, "
        f"Val Loss = {history['val_loss'][ep]:.4f}, "
        f"Val Acc = {history['val_acc'][ep]:.2f}%, "
        f"LR = {history['lr'][ep]:.6f}\n"
    )

results_text += f"""
6. BEST CHECKPOINT SUMMARY:
- Best Epoch:             {best_epoch}
- Best Validation Loss:   {best_val_loss:.4f}
- Best Validation Acc:    {best_val_acc*100:.2f}%
- Training Duration:      {total_training_duration}s ({round(total_training_duration/60, 1)} minutes)
- Best Model Path:        {BEST_MODEL_PATH}
- Final Model Path:       {FINAL_MODEL_PATH}

PRODUCTION INTEGRITY STATUS:
- Production baseline model (NetraSetu_ResNet50_best.pth) remained 100% UNTOUCHED.
"""

with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
    f.write(results_text)
print(f"Saved full training results to: {RESULTS_TXT_PATH}")
print("\n[SUCCESS] RESNET-50 COMBINED TRAINING COMPLETED.")

if __name__ == "__main__":
    pass
