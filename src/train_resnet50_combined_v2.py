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
# NETRASETU - COMBINED RESNET-50 V2 (QUALITY-FILTERED TRAINING)
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "combined_grading_v2")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "combined_training_v2")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "NetraSetu_ResNet50_combined_v2_best.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "NetraSetu_ResNet50_combined_v2_final.pth")

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
print("NETRASETU COMBINED RESNET-50 V2 (QUALITY-FILTERED)")
print("==================================================")
print(f"PyTorch Version: {torch.__version__}")
print(f"Device:          {device}")

if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    total_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    print(f"GPU Total VRAM:  {total_mem} GB")
    use_amp = True
    scaler = torch.amp.GradScaler('cuda')
else:
    total_mem = 0
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
# 4. DATASET & DATALOADERS
# ------------------------------------------------------------
class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root)
        self.alb_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        image_np = np.array(image)

        if self.alb_transform is not None:
            augmented = self.alb_transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, target

print("\nLoading datasets...")
train_dataset = AlbumentationsDataset(TRAIN_DIR, transform=train_transform)
val_dataset = AlbumentationsDataset(VAL_DIR, transform=val_transform)

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

print(f"  Training Samples:   {len(train_dataset)}")
print(f"  Validation Samples: {len(val_dataset)}")
print(f"  Class Names:        {train_dataset.classes}")

# ------------------------------------------------------------
# 5. CLASS WEIGHTS (Calculated from TRAINING SPLIT ONLY)
# ------------------------------------------------------------
train_targets = [s[1] for s in train_dataset.samples]
class_counts = np.bincount(train_targets, minlength=5)
n_total = len(train_targets)

# weight_c = N_total / (5 * N_c)
class_weights = n_total / (5.0 * class_counts.astype(np.float32))
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("\nV2 Training Split Class Distribution:")
for c in range(5):
    print(f"  Class {c}: {class_counts[c]} images | Weight: {class_weights[c]:.4f}")

# ------------------------------------------------------------
# 6. MODEL SETUP (Torchvision ResNet-50)
# ------------------------------------------------------------
print("\nInitializing ResNet-50 with pretrained ImageNet weights...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Replace final fully connected layer for 5 DR classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

# ------------------------------------------------------------
# 7. TRAINING LOOP
# ------------------------------------------------------------
best_val_loss = float("inf")
best_val_acc = 0.0
best_epoch = 0
best_model_wts = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": []
}

print("\n" + "="*50)
print("STARTING 12-EPOCH TRAINING RUN (V2 QUALITY-FILTERED)")
print("="*50)
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # --- TRAIN PHASE ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False)
    for inputs, labels in train_pbar:
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

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data).item()
        total_train += inputs.size(0)

        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = (correct_train / total_train) * 100.0

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]", leave=False)
        for inputs, labels in val_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data).item()
            total_val += inputs.size(0)

    epoch_val_loss = val_loss / total_val
    epoch_val_acc = (correct_val / total_val) * 100.0

    # Scheduler Step
    scheduler.step(epoch_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)
    history["lr"].append(current_lr)

    epoch_duration = time.time() - epoch_start
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_duration:.1f}s"
    )

    # Checkpoint on lowest validation loss
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_val_acc = epoch_val_acc / 100.0
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": best_val_loss,
            "val_accuracy": best_val_acc,
            "class_weights": weights_tensor.cpu().numpy(),
            "classes": train_dataset.classes
        }, BEST_MODEL_PATH)
        print(f"  --> Saved NEW best checkpoint (Epoch {epoch}, Val Loss: {best_val_loss:.4f}) to: {BEST_MODEL_PATH}")

total_training_duration = round(time.time() - start_time, 1)
print(f"\nTraining completed in {total_training_duration}s ({round(total_training_duration / 60, 1)} min).")
print(f"Best Validation Loss: {best_val_loss:.4f} (Accuracy: {best_val_acc*100:.2f}%) at Epoch {best_epoch}")

# ------------------------------------------------------------
# 8. SAVE FINAL MODEL
# ------------------------------------------------------------
torch.save({
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "val_loss": history["val_loss"][-1],
    "val_accuracy": history["val_acc"][-1] / 100.0,
    "class_weights": weights_tensor.cpu().numpy(),
    "classes": train_dataset.classes
}, FINAL_MODEL_PATH)
print(f"Saved FINAL combined V2 checkpoint to: {FINAL_MODEL_PATH}")

# ------------------------------------------------------------
# 9. PLOT & SAVE TRAINING CURVES
# ------------------------------------------------------------
epochs_range = range(1, EPOCHS + 1)

# Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
plt.plot(epochs_range, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
plt.title("NetraSetu ResNet-50 Combined V2: Training vs Validation Loss", fontsize=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Weighted CrossEntropy Loss", fontsize=11)
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
plt.title("NetraSetu ResNet-50 Combined V2: Training vs Validation Accuracy", fontsize=12)
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
    "experiment": "Combined V2 (Quality-Filtered)",
    "dataset": "APTOS (2161) + IDRiD GOOD (320) + IDRiD REVIEW (81) = 2562 total",
    "idrid_good_used": 320,
    "idrid_review_used": 81,
    "idrid_reject_excluded": 12,
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

results_text = f"""NETRASETU - RESNET-50 COMBINED V2 TRAINING REPORT
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

1. SYSTEM & DEVICE:
- Device:                 {device}
- GPU Model:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
- CUDA Memory:            {total_mem} GB
- Mixed Precision (AMP):  {use_amp}

2. DATASET CONFIGURATION:
- APTOS Train Images:     2161 (accepted quality)
- IDRiD GOOD Used:        320
- IDRiD REVIEW Used:      81
- IDRiD REJECT Excluded:  12 (severely blurred images excluded)
- Total Development Pool: {len(train_dataset) + len(val_dataset)} images (2562)
- Training Split Size:    {len(train_dataset)} images (85.0%)
- Validation Split Size:  {len(val_dataset)} images (15.0%)
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
- Previous combined V1 model (NetraSetu_ResNet50_combined_best.pth) remained 100% UNTOUCHED.
"""

with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as f:
    f.write(results_text)
print(f"Saved full training results to: {RESULTS_TXT_PATH}")
print("\n[SUCCESS] RESNET-50 COMBINED V2 TRAINING COMPLETED.")

if __name__ == "__main__":
    pass
