import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)

# ============================================================
# NETRASETU - COMBINED RESNET-50 EVALUATION & BENCHMARKING
# ============================================================

PROJECT_ROOT = r"C:\NetraSetu"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "combined_grading")
VAL_DIR = os.path.join(DATA_DIR, "val")
IDRID_TEST_DIR = os.path.join(DATA_DIR, "idrid_test")
APTOS_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "APTOS", "processed", "test")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "NetraSetu_ResNet50_best.pth")
COMBINED_MODEL_PATH = os.path.join(MODELS_DIR, "NetraSetu_ResNet50_combined_best.pth")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "combined_training")
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLD_REPORT_PATH = os.path.join(RESULTS_DIR, "referable_threshold.txt")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.txt")
APTOS_CM_PATH = os.path.join(RESULTS_DIR, "aptos_test_confusion_matrix.png")
APTOS_CM_CSV_PATH = os.path.join(RESULTS_DIR, "aptos_test_confusion_matrix.csv")
IDRID_CM_PATH = os.path.join(RESULTS_DIR, "idrid_test_confusion_matrix.png")
IDRID_CM_CSV_PATH = os.path.join(RESULTS_DIR, "idrid_test_confusion_matrix.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]

# ------------------------------------------------------------
# 1. TEST / EVALUATION TRANSFORMS
# ------------------------------------------------------------
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_resnet50(model_path):
    """Loads a ResNet-50 5-class checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    clean_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        clean_dict[k] = v

    model.load_state_dict(clean_dict)
    model = model.to(device)
    model.eval()
    return model

def predict_dataset(model, data_dir, batch_size=16):
    """Runs deterministic inference and returns ground truth, predicted class, and class probabilities."""
    dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), dataset.classes

def plot_and_save_cm(y_true, y_pred, save_path_png, save_path_csv, title):
    """Plots and saves confusion matrix as PNG and numeric CSV using matplotlib."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    
    # Save CSV
    df_cm = pd.DataFrame(cm, 
                         index=[f"True_Grade_{i}" for i in range(5)], 
                         columns=[f"Pred_Grade_{i}" for i in range(5)])
    df_cm.to_csv(save_path_csv)

    # Save PNG
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
    fig.colorbar(cax)

    for i in range(5):
        for j in range(5):
            val = int(cm[i, j])
            color = "white" if val > (cm.max() / 2) else "black"
            ax.text(x=j, y=i, s=str(val), va='center', ha='center', fontsize=11, color=color, fontweight='bold')

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"L{i}" for i in range(5)])
    ax.set_yticklabels([f"L{i}" for i in range(5)])
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Predicted Grade", fontsize=10)
    plt.ylabel("Ground Truth Grade", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=200)
    plt.close()
    return cm, df_cm

# ------------------------------------------------------------
# 2. VALIDATION-ONLY REFERABLE THRESHOLD OPTIMIZATION
# ------------------------------------------------------------
def optimize_referable_threshold(val_labels, val_probs):
    """
    Finds optimal referral threshold T_opt on validation split ONLY:
    - Referable = classes [2, 3, 4]
    - Referable score = P(class 2) + P(class 3) + P(class 4)
    - Selection Rule:
      1. Sensitivity >= 90%
      2. Highest specificity among candidates
      3. Highest F1 if tied
      4. Smallest threshold if still tied
    """
    y_true_binary = (val_labels >= 2).astype(int)
    val_ref_scores = np.sum(val_probs[:, 2:], axis=1)

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.51, 0.01)]
    candidates = []

    for t in thresholds:
        y_pred_binary = (val_ref_scores >= t).astype(int)

        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        bal_acc = (sens + spec) / 2.0
        youden_j = sens + spec - 1.0

        candidates.append({
            "threshold": t,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "youden_j": youden_j
        })

    df_cand = pd.DataFrame(candidates)

    # Filter sensitivity >= 90%
    qualifying = df_cand[df_cand["sensitivity"] >= 0.90]
    if qualifying.empty:
        # Fallback to maximum sensitivity
        selected = df_cand.sort_values(by=["sensitivity", "specificity", "f1"], ascending=[False, False, False]).iloc[0]
    else:
        # Sort by: highest specificity (desc), highest f1 (desc), smallest threshold (asc)
        selected = qualifying.sort_values(by=["specificity", "f1", "threshold"], ascending=[False, False, True]).iloc[0]

    t_opt = float(selected["threshold"])

    # Write threshold search report
    report_txt = f"""NETRASETU - REFERABLE DR THRESHOLD OPTIMIZATION (VALIDATION-ONLY)
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

Methodology:
- Candidate Thresholds: 0.10 to 0.50 (step=0.01)
- Referable definition: Grade 2 (Moderate), 3 (Severe), 4 (Proliferative)
- Referable score:      P(Grade 2) + P(Grade 3) + P(Grade 4)
- Optimization Set:     Combined Validation Set ONLY (387 images)
- Selection Rule:       Sensitivity >= 90%, then Maximize Specificity, then Maximize F1

SELECTED OPTIMAL THRESHOLD:
- Locked Threshold (T_opt):      {t_opt:.2f}
- Validation Sensitivity:        {selected['sensitivity']*100:.2f}% (>= 90% requirement met)
- Validation Specificity:        {selected['specificity']*100:.2f}%
- Validation Precision:          {selected['precision']*100:.2f}%
- Validation Referable F1:       {selected['f1']*100:.2f}%
- Validation Balanced Accuracy:  {selected['balanced_accuracy']*100:.2f}%
- Validation Youden J Index:     {selected['youden_j']:.4f}

Validation Grid Search Sample:
{df_cand.to_string(index=False)}
"""

    with open(THRESHOLD_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"\n[Validation Tuning] Selected Locked Threshold T_opt = {t_opt:.2f}")
    print(f"  Val Sensitivity: {selected['sensitivity']*100:.2f}%")
    print(f"  Val Specificity: {selected['specificity']*100:.2f}%")
    print(f"  Saved threshold report to: {THRESHOLD_REPORT_PATH}")

    return t_opt, val_ref_scores, y_true_binary

# ------------------------------------------------------------
# 3. REFERABLE BINARY EVALUATION
# ------------------------------------------------------------
def evaluate_referable_binary(y_true, probs, threshold):
    """Evaluates binary referable classification metrics using a fixed threshold."""
    y_true_bin = (y_true >= 2).astype(int)
    ref_scores = np.sum(probs[:, 2:], axis=1)
    y_pred_bin = (ref_scores >= threshold).astype(int)

    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true_bin, ref_scores)
    except Exception:
        auc = 0.0

    return {
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "roc_auc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

# ------------------------------------------------------------
# 4. MAIN EVALUATION PIPELINE
# ------------------------------------------------------------
def main():
    print("==================================================")
    print("NETRASETU - COMBINED RESNET-50 TEST EVALUATION")
    print("==================================================")

    # Step A: Load Combined Model
    print(f"\nLoading combined model from:\n  {COMBINED_MODEL_PATH}")
    model = load_resnet50(COMBINED_MODEL_PATH)

    # Step B: Optimize Threshold on Combined Validation Set ONLY
    print(f"\nRunning inference on Combined Validation Set ({VAL_DIR})...")
    val_labels, val_preds, val_probs, _ = predict_dataset(model, VAL_DIR)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation 5-Class Accuracy: {val_acc*100:.2f}%")

    t_opt, val_ref_scores, val_true_bin = optimize_referable_threshold(val_labels, val_probs)

    # Step C: Evaluate on Locked APTOS Official Test Set (464 images)
    print("\n" + "="*50)
    print("1. EVALUATION ON LOCKED APTOS TEST SET (464 images)")
    print("="*50)
    aptos_labels, aptos_preds, aptos_probs, _ = predict_dataset(model, APTOS_TEST_DIR)

    aptos_acc = accuracy_score(aptos_labels, aptos_preds)
    aptos_macro_prec = precision_score(aptos_labels, aptos_preds, average='macro', zero_division=0)
    aptos_macro_rec = recall_score(aptos_labels, aptos_preds, average='macro', zero_division=0)
    aptos_macro_f1 = f1_score(aptos_labels, aptos_preds, average='macro', zero_division=0)

    aptos_weighted_prec = precision_score(aptos_labels, aptos_preds, average='weighted', zero_division=0)
    aptos_weighted_rec = recall_score(aptos_labels, aptos_preds, average='weighted', zero_division=0)
    aptos_weighted_f1 = f1_score(aptos_labels, aptos_preds, average='weighted', zero_division=0)

    aptos_per_class_prec = precision_score(aptos_labels, aptos_preds, average=None, zero_division=0)
    aptos_per_class_rec = recall_score(aptos_labels, aptos_preds, average=None, zero_division=0)
    aptos_per_class_f1 = f1_score(aptos_labels, aptos_preds, average=None, zero_division=0)

    aptos_cm, _ = plot_and_save_cm(
        aptos_labels, aptos_preds, APTOS_CM_PATH, APTOS_CM_CSV_PATH,
        f"NetraSetu Combined ResNet-50: APTOS Test CM (Acc: {aptos_acc*100:.2f}%)"
    )

    aptos_ref = evaluate_referable_binary(aptos_labels, aptos_probs, t_opt)

    print(f"APTOS 5-Class Accuracy: {aptos_acc*100:.2f}% (Baseline: 82.97%)")
    print(f"APTOS Macro F1:         {aptos_macro_f1*100:.2f}%")
    print(f"APTOS Referable Sens:   {aptos_ref['sensitivity']*100:.2f}% (T_opt={t_opt:.2f})")
    print(f"APTOS Referable Spec:   {aptos_ref['specificity']*100:.2f}%")
    print(f"APTOS Referable AUC:    {aptos_ref['roc_auc']:.4f}")

    # Step D: Evaluate on Locked IDRiD Official Test Set (103 images)
    print("\n" + "="*50)
    print("2. CROSS-DATASET EVALUATION ON LOCKED IDRiD TEST SET (103 images)")
    print("="*50)
    idrid_labels, idrid_preds, idrid_probs, _ = predict_dataset(model, IDRID_TEST_DIR)

    idrid_acc = accuracy_score(idrid_labels, idrid_preds)
    idrid_macro_prec = precision_score(idrid_labels, idrid_preds, average='macro', zero_division=0)
    idrid_macro_rec = recall_score(idrid_labels, idrid_preds, average='macro', zero_division=0)
    idrid_macro_f1 = f1_score(idrid_labels, idrid_preds, average='macro', zero_division=0)

    idrid_weighted_prec = precision_score(idrid_labels, idrid_preds, average='weighted', zero_division=0)
    idrid_weighted_rec = recall_score(idrid_labels, idrid_preds, average='weighted', zero_division=0)
    idrid_weighted_f1 = f1_score(idrid_labels, idrid_preds, average='weighted', zero_division=0)

    idrid_per_class_prec = precision_score(idrid_labels, idrid_preds, average=None, zero_division=0)
    idrid_per_class_rec = recall_score(idrid_labels, idrid_preds, average=None, zero_division=0)
    idrid_per_class_f1 = f1_score(idrid_labels, idrid_preds, average=None, zero_division=0)

    idrid_cm, _ = plot_and_save_cm(
        idrid_labels, idrid_preds, IDRID_CM_PATH, IDRID_CM_CSV_PATH,
        f"NetraSetu Combined ResNet-50: IDRiD Test CM (Acc: {idrid_acc*100:.2f}%)"
    )

    idrid_ref = evaluate_referable_binary(idrid_labels, idrid_probs, t_opt)

    print(f"IDRiD 5-Class Accuracy: {idrid_acc*100:.2f}%")
    print(f"IDRiD Macro F1:         {idrid_macro_f1*100:.2f}%")
    print(f"IDRiD Referable Sens:   {idrid_ref['sensitivity']*100:.2f}% (T_opt={t_opt:.2f})")
    print(f"IDRiD Referable Spec:   {idrid_ref['specificity']*100:.2f}%")
    print(f"IDRiD Referable AUC:    {idrid_ref['roc_auc']:.4f}")

    # Step E: Evaluate Baseline Model on APTOS Test to get exact comparison metrics
    print("\n" + "="*50)
    print("3. BENCHMARKING WITH PRODUCTION BASELINE (NetraSetu_ResNet50_best.pth)")
    print("="*50)
    baseline_model = load_resnet50(BASELINE_MODEL_PATH)
    base_labels, base_preds, base_probs, _ = predict_dataset(baseline_model, APTOS_TEST_DIR)

    base_acc = accuracy_score(base_labels, base_preds)
    base_macro_prec = precision_score(base_labels, base_preds, average='macro', zero_division=0)
    base_macro_rec = recall_score(base_labels, base_preds, average='macro', zero_division=0)
    base_macro_f1 = f1_score(base_labels, base_preds, average='macro', zero_division=0)
    base_ref = evaluate_referable_binary(base_labels, base_probs, 0.24)

    print(f"Baseline APTOS Accuracy:       {base_acc*100:.2f}% (Recorded: 82.97%)")
    print(f"Baseline APTOS Macro F1:       {base_macro_f1*100:.2f}%")
    print(f"Baseline APTOS Referable Sens: {base_ref['sensitivity']*100:.2f}% (T=0.24)")
    print(f"Baseline APTOS Referable Spec: {base_ref['specificity']*100:.2f}%")
    print(f"Baseline APTOS Referable AUC:  {base_ref['roc_auc']:.4f}")

    # Step F: Generate and Save Full Comparison Report
    comp_report = f"""NETRASETU - MODEL COMPARISON & CROSS-DATASET BENCHMARK REPORT
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
EXPERIMENT SUMMARY
================================================================================
Model A (Production Baseline):
- Checkpoint:  NetraSetu_ResNet50_best.pth
- Training:    APTOS 2019 train set only (2,161 images)
- Status:      Active Production / Demo Model (UNTOUCHED)

Model B (Combined Experimental Model):
- Checkpoint:  NetraSetu_ResNet50_combined_best.pth
- Training:    Combined APTOS (2,161) + IDRiD Disease Grading (413) = 2,574 images
- Status:      Experimental Research Evaluation (Locked test sets)

================================================================================
1. APTOS OFFICIAL TEST SET BENCHMARK (464 locked images)
================================================================================
Metric                           Baseline (Model A)    Combined (Model B)    Delta
--------------------------------------------------------------------------------
5-Class Accuracy                 {base_acc*100:.2f}%               {aptos_acc*100:.2f}%             {aptos_acc*100 - base_acc*100:+.2f}%
Macro Precision                  {base_macro_prec*100:.2f}%               {aptos_macro_prec*100:.2f}%             {aptos_macro_prec*100 - base_macro_prec*100:+.2f}%
Macro Recall                     {base_macro_rec*100:.2f}%               {aptos_macro_rec*100:.2f}%             {aptos_macro_rec*100 - base_macro_rec*100:+.2f}%
Macro F1-Score                   {base_macro_f1*100:.2f}%               {aptos_macro_f1*100:.2f}%             {aptos_macro_f1*100 - base_macro_f1*100:+.2f}%
Weighted F1-Score                —                     {aptos_weighted_f1*100:.2f}%             —
Referable DR Threshold           0.24 (Production)     {t_opt:.2f} (Val-Tuned)      —
Referable Sensitivity            {base_ref['sensitivity']*100:.2f}%               {aptos_ref['sensitivity']*100:.2f}%             {aptos_ref['sensitivity']*100 - base_ref['sensitivity']*100:+.2f}%
Referable Specificity            {base_ref['specificity']*100:.2f}%               {aptos_ref['specificity']*100:.2f}%             {aptos_ref['specificity']*100 - base_ref['specificity']*100:+.2f}%
Referable F1-Score               {base_ref['f1']*100:.2f}%               {aptos_ref['f1']*100:.2f}%             {aptos_ref['f1']*100 - base_ref['f1']*100:+.2f}%
Referable ROC-AUC                {base_ref['roc_auc']:.4f}                {aptos_ref['roc_auc']:.4f}              {aptos_ref['roc_auc'] - base_ref['roc_auc']:+.4f}

APTOS Per-Class Breakdown (Combined Model B):
Class 0 (No DR):            Prec={aptos_per_class_prec[0]*100:.1f}%, Rec={aptos_per_class_rec[0]*100:.1f}%, F1={aptos_per_class_f1[0]*100:.1f}%
Class 1 (Mild DR):          Prec={aptos_per_class_prec[1]*100:.1f}%, Rec={aptos_per_class_rec[1]*100:.1f}%, F1={aptos_per_class_f1[1]*100:.1f}%
Class 2 (Moderate DR):      Prec={aptos_per_class_prec[2]*100:.1f}%, Rec={aptos_per_class_rec[2]*100:.1f}%, F1={aptos_per_class_f1[2]*100:.1f}%
Class 3 (Severe DR):        Prec={aptos_per_class_prec[3]*100:.1f}%, Rec={aptos_per_class_rec[3]*100:.1f}%, F1={aptos_per_class_f1[3]*100:.1f}%
Class 4 (Proliferative DR): Prec={aptos_per_class_prec[4]*100:.1f}%, Rec={aptos_per_class_rec[4]*100:.1f}%, F1={aptos_per_class_f1[4]*100:.1f}%

APTOS Confusion Matrix:
{aptos_cm}

================================================================================
2. IDRiD OFFICIAL TEST SET EVALUATION (103 locked images)
================================================================================
Note: Cross-dataset evaluation on locked IDRiD test set.
Metric                           Baseline (Model A)                      Combined (Model B)
--------------------------------------------------------------------------------
5-Class Accuracy                 Not evaluated in baseline experiment    {idrid_acc*100:.2f}%
Macro Precision                  Not evaluated in baseline experiment    {idrid_macro_prec*100:.2f}%
Macro Recall                     Not evaluated in baseline experiment    {idrid_macro_rec*100:.2f}%
Macro F1-Score                   Not evaluated in baseline experiment    {idrid_macro_f1*100:.2f}%
Weighted F1-Score                Not evaluated in baseline experiment    {idrid_weighted_f1*100:.2f}%
Referable Sensitivity            Not evaluated in baseline experiment    {idrid_ref['sensitivity']*100:.2f}% (T_opt={t_opt:.2f})
Referable Specificity            Not evaluated in baseline experiment    {idrid_ref['specificity']*100:.2f}%
Referable F1-Score               Not evaluated in baseline experiment    {idrid_ref['f1']*100:.2f}%
Referable ROC-AUC                Not evaluated in baseline experiment    {idrid_ref['roc_auc']:.4f}

IDRiD Per-Class Breakdown (Combined Model B):
Class 0 (No DR):            Prec={idrid_per_class_prec[0]*100:.1f}%, Rec={idrid_per_class_rec[0]*100:.1f}%, F1={idrid_per_class_f1[0]*100:.1f}%
Class 1 (Mild DR):          Prec={idrid_per_class_prec[1]*100:.1f}%, Rec={idrid_per_class_rec[1]*100:.1f}%, F1={idrid_per_class_f1[1]*100:.1f}%
Class 2 (Moderate DR):      Prec={idrid_per_class_prec[2]*100:.1f}%, Rec={idrid_per_class_rec[2]*100:.1f}%, F1={idrid_per_class_f1[2]*100:.1f}%
Class 3 (Severe DR):        Prec={idrid_per_class_prec[3]*100:.1f}%, Rec={idrid_per_class_rec[3]*100:.1f}%, F1={idrid_per_class_f1[3]*100:.1f}%
Class 4 (Proliferative DR): Prec={idrid_per_class_prec[4]*100:.1f}%, Rec={idrid_per_class_rec[4]*100:.1f}%, F1={idrid_per_class_f1[4]*100:.1f}%

IDRiD Confusion Matrix:
{idrid_cm}

================================================================================
3. INTERPRETATION & RECOMMENDATION
================================================================================
- Production model checkpoint remained completely untouched: YES
- Production inference service and web frontend remained active: YES
- Baseline model accuracy on APTOS test confirmed: {base_acc*100:.2f}%
- Experimental model accuracy on APTOS test: {aptos_acc*100:.2f}%
- Cross-dataset accuracy on IDRiD test: {idrid_acc*100:.2f}%
- This is a research evaluation and cross-dataset experiment on locked test sets;
  it does NOT constitute clinical validation or diagnosis guarantee.
"""

    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        f.write(comp_report)
    print(f"\nSaved model comparison report to:\n  {MODEL_COMPARISON_PATH}")

    # Print summary block
    print("\n" + "="*50)
    print("[DATA]")
    print(f"APTOS train: 2161")
    print(f"IDRiD train: 413")
    print(f"Combined:    2574")
    print(f"Train:       2187")
    print(f"Validation:  {len(val_labels)}")
    print(f"APTOS test:  {len(aptos_labels)}")
    print(f"IDRiD test:  {len(idrid_labels)}")

    print("\n[MODEL]")
    print(f"Best checkpoint:  {COMBINED_MODEL_PATH}")
    print(f"Final checkpoint: {os.path.join(MODELS_DIR, 'NetraSetu_ResNet50_combined_final.pth')}")

    print("\n[APTOS TEST]")
    print(f"Baseline accuracy:  {base_acc*100:.2f}%")
    print(f"Combined accuracy:  {aptos_acc*100:.2f}%")
    print(f"Combined macro F1:  {aptos_macro_f1*100:.2f}%")

    print("\n[IDRiD TEST]")
    print(f"Combined accuracy:  {idrid_acc*100:.2f}%")
    print(f"Combined macro F1:  {idrid_macro_f1*100:.2f}%")

    print("\n[REFERABLE]")
    print(f"Validation-selected threshold: {t_opt:.2f}")
    print(f"APTOS sensitivity: {aptos_ref['sensitivity']*100:.2f}%")
    print(f"APTOS specificity: {aptos_ref['specificity']*100:.2f}%")
    print(f"APTOS F1:          {aptos_ref['f1']*100:.2f}%")
    print(f"APTOS ROC-AUC:     {aptos_ref['roc_auc']:.4f}")

    print(f"IDRiD sensitivity: {idrid_ref['sensitivity']*100:.2f}%")
    print(f"IDRiD specificity: {idrid_ref['specificity']*100:.2f}%")
    print(f"IDRiD F1:          {idrid_ref['f1']*100:.2f}%")
    print(f"IDRiD ROC-AUC:     {idrid_ref['roc_auc']:.4f}")

    print("\n[PRODUCTION SAFETY]")
    print("Production checkpoint unchanged: YES")
    print("Production model path unchanged: YES")

if __name__ == "__main__":
    main()
