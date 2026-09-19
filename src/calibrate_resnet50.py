"""
src/calibrate_resnet50.py
Confidence Calibration for NetraSetu Production Baseline ResNet-50 using Temperature Scaling.

Requirements:
1. Uses production model: models/NetraSetu_ResNet50_best.pth (DOES NOT MODIFY WEIGHTS).
2. Uses validation split only: data/APTOS/processed/val (463 images).
3. Fits a single positive temperature T via NLL minimization.
4. Calculates NLL, ECE, Brier score before and after calibration.
5. Saves:
    results/final_pipeline/calibration/temperature_scaling.json
    results/final_pipeline/calibration/reliability_before.png
    results/final_pipeline/calibration/reliability_after.png
6. Production threshold remains locked at 0.24.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    probs: (N, C) numpy array
    labels: (N,) numpy array
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_data.append({
                'bin': i,
                'lower': bin_lower,
                'upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': int(np.sum(in_bin))
            })
        else:
            bin_data.append({
                'bin': i,
                'lower': bin_lower,
                'upper': bin_upper,
                'accuracy': 0.0,
                'confidence': (bin_lower + bin_upper) / 2.0,
                'count': 0
            })

    return float(ece), bin_data

def compute_brier_score(probs, labels, num_classes=5):
    """
    Computes multi-class Brier score.
    """
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    for i, l in enumerate(labels):
        one_hot[i, l] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def plot_reliability_diagram(bin_data, ece, title, save_path):
    plt.figure(figsize=(6, 5), dpi=150)
    centers = [(b['lower'] + b['upper']) / 2.0 for b in bin_data]
    accs = [b['accuracy'] for b in bin_data]
    confs = [b['confidence'] for b in bin_data]
    counts = [b['count'] for b in bin_data]

    plt.bar(centers, accs, width=0.08, alpha=0.7, color='#2563eb', edgecolor='#1d4ed8', label='Observed Accuracy')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#ef4444', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f"{title}\nECE = {ece:.4f}", fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calibrate():
    print("=" * 60)
    print("NETRASETU — RESNET-50 CONFIDENCE CALIBRATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"C:\NetraSetu\models\NetraSetu_ResNet50_best.pth"
    val_dir = r"C:\NetraSetu\data\APTOS\processed\val"
    out_dir = r"C:\NetraSetu\results\final_pipeline\calibration"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded production baseline from {model_path} (Strictly in Eval mode, no weights modified)")

    # 2. Extract Logits and Labels
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Loaded validation set: {len(val_dataset)} images across classes: {val_dataset.classes}")

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits_tensor = torch.cat(all_logits, dim=0) # (463, 5)
    labels_tensor = torch.cat(all_labels, dim=0) # (463,)
    print(f"Extracted logits tensor: {logits_tensor.shape}, labels: {labels_tensor.shape}")

    # 3. Before Calibration Metrics
    criterion = nn.CrossEntropyLoss()
    nll_before = float(criterion(logits_tensor, labels_tensor).item())
    probs_before = torch.softmax(logits_tensor, dim=1).numpy()
    labels_np = labels_tensor.numpy()

    ece_before, bin_data_before = compute_ece(probs_before, labels_np, n_bins=10)
    brier_before = compute_brier_score(probs_before, labels_np, num_classes=5)

    print("\nBEFORE CALIBRATION (T = 1.0):")
    print(f"  NLL (CrossEntropy): {nll_before:.4f}")
    print(f"  ECE:                {ece_before:.4f}")
    print(f"  Brier Score:        {brier_before:.4f}")

    # 4. Temperature Optimization
    # We parameterize temperature as T = log_temp.exp() to ensure T > 0 strictly
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = optim.LBFGS([log_temperature], lr=0.01, max_iter=100)

    def eval_loss():
        optimizer.zero_grad()
        T = log_temperature.exp()
        scaled_logits = logits_tensor / T
        loss = criterion(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    optimal_T = float(log_temperature.exp().item())
    print(f"\nOPTIMIZATION CONVERGED:")
    print(f"  Optimal Temperature T: {optimal_T:.4f}")

    # 5. After Calibration Metrics
    scaled_logits_tensor = logits_tensor / optimal_T
    nll_after = float(criterion(scaled_logits_tensor, labels_tensor).item())
    probs_after = torch.softmax(scaled_logits_tensor, dim=1).numpy()

    ece_after, bin_data_after = compute_ece(probs_after, labels_np, n_bins=10)
    brier_after = compute_brier_score(probs_after, labels_np, num_classes=5)

    print("\nAFTER TEMPERATURE CALIBRATION:")
    print(f"  NLL:         {nll_after:.4f} (Delta: {nll_after - nll_before:+.4f})")
    print(f"  ECE:         {ece_after:.4f} (Delta: {ece_after - ece_before:+.4f})")
    print(f"  Brier Score: {brier_after:.4f} (Delta: {brier_after - brier_before:+.4f})")

    # 6. Save Artifacts
    calib_json_path = os.path.join(out_dir, "temperature_scaling.json")
    calib_data = {
        'model_name': 'NetraSetu_ResNet50_best.pth',
        'validation_dataset': 'APTOS_processed_val',
        'validation_samples': len(val_dataset),
        'optimal_temperature': round(optimal_T, 4),
        'metrics_before': {
            'NLL': round(nll_before, 4),
            'ECE': round(ece_before, 4),
            'Brier': round(brier_before, 4)
        },
        'metrics_after': {
            'NLL': round(nll_after, 4),
            'ECE': round(ece_after, 4),
            'Brier': round(brier_after, 4)
        },
        'locked_production_threshold': 0.24,
        'status': 'READY'
    }
    with open(calib_json_path, 'w', encoding='utf-8') as f:
        json.dump(calib_data, f, indent=2)
    print(f"Saved calibration parameters to {calib_json_path}")

    # 7. Generate Reliability Diagrams
    rel_before_path = os.path.join(out_dir, "reliability_before.png")
    rel_after_path = os.path.join(out_dir, "reliability_after.png")
    plot_reliability_diagram(bin_data_before, ece_before, "Reliability Diagram (Before Calibration, T=1.0)", rel_before_path)
    plot_reliability_diagram(bin_data_after, ece_after, f"Reliability Diagram (After Calibration, T={optimal_T:.2f})", rel_after_path)
    print(f"Saved reliability diagrams to {rel_before_path} and {rel_after_path}")

if __name__ == '__main__':
    calibrate()
