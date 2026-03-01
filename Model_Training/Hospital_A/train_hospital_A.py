"""
===============================================================================
 HOSPITAL A - LOCAL MODEL TRAINING (5-FOLD CROSS-VALIDATION)
===============================================================================

This script trains the LSTM model using ONLY Hospital A's local data.

Pipeline:
  1. Load X.npy and y.npy
  2. Run 5-Fold Stratified CV to find best hyperparams & threshold
  3. Retrain final model on ALL 41 patients using best config
  4. Save final model weights as local_model.pth

Hospital A Details:
  - ~41 patients
  - ~12 mortality cases, ~29 survived
  - Mortality rate: ~29%

===============================================================================
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json

# Add parent paths to import LSTM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from LSTM import MortalityLSTM

# ============================================================
# CONFIGURATION
# ============================================================
HOSPITAL_NAME = "A"
HOSPITAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_EPOCHS = 300             # More epochs for better convergence
LOCAL_LR = 0.002               # Learning rate
BATCH_SIZE = 8                 # Batch size for local data
RANDOM_SEED = 42
N_FOLDS = 5                   # Cross-validation folds

# ============================================================
# LOAD LOCAL DATA
# ============================================================
def load_hospital_data(hospital_dir):
    """
    Loads Hospital A's local dataset from .npy files.
    """
    X_path = os.path.join(hospital_dir, 'X.npy')
    y_path = os.path.join(hospital_dir, 'y.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Dataset files not found in {hospital_dir}")
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"  X shape: {X.shape} (patients, max_days, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Survived: {int(len(y) - y.sum())}  |  Died: {int(y.sum())}  |  Mortality: {(y.sum()/len(y))*100:.1f}%\n")
    
    return X, y


# ============================================================
# TRAIN ONE FOLD
# ============================================================
def train_one_fold(X_train, y_train, epochs=LOCAL_EPOCHS, lr=LOCAL_LR):
    """
    Trains LSTM on one fold's training data.
    Returns trained model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MortalityLSTM().to(device)
    model.train()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    
    # Class-weighted loss
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    
    return model


# ============================================================
# EVALUATE ONE FOLD
# ============================================================
def evaluate_fold(model, X_test, y_test):
    """
    Evaluates model on fold's test data with optimal threshold.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_t).cpu().numpy().flatten()
        y_proba = 1 / (1 + np.exp(-logits))
        y_true = y_test.astype(int)
    
    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.2, 0.8, 0.05):
        f1_temp = f1_score(y_true, (y_proba > thresh).astype(int), zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_thresh = thresh
    
    y_pred = (y_proba > best_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0
    
    return {'accuracy': acc, 'auc_roc': auc, 'f1_score': f1, 'threshold': best_thresh}


# ============================================================
# MAIN: 5-FOLD CV + FINAL RETRAIN
# ============================================================
if __name__ == "__main__":
    
    print("=" * 70)
    print(f"  HOSPITAL {HOSPITAL_NAME} - LOCAL TRAINING ({N_FOLDS}-FOLD CV)")
    print("=" * 70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")
    
    # Load local data
    X_local, y_local = load_hospital_data(HOSPITAL_DIR)
    
    # ============================================================
    # PHASE 1: 5-FOLD CROSS-VALIDATION
    # ============================================================
    print("─" * 70)
    print(f"  PHASE 1: {N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
    print("─" * 70 + "\n")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_metrics = []
    best_fold_auc = -1
    best_fold_idx = 0
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_local, y_local)):
        X_train, X_val = X_local[train_idx], X_local[val_idx]
        y_train, y_val = y_local[train_idx], y_local[val_idx]
        
        print(f"  Fold {fold_idx+1}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)} ", end="")
        
        # Train
        model = train_one_fold(X_train, y_train, epochs=LOCAL_EPOCHS, lr=LOCAL_LR)
        
        # Evaluate on validation set
        metrics = evaluate_fold(model, X_val, y_val)
        fold_metrics.append(metrics)
        
        print(f"→ Acc={metrics['accuracy']:.4f}  AUC={metrics['auc_roc']:.4f}  F1={metrics['f1_score']:.4f}")
        
        # Track best fold by AUC
        if metrics['auc_roc'] > best_fold_auc:
            best_fold_auc = metrics['auc_roc']
            best_fold_idx = fold_idx
    
    # Average metrics across folds
    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
    avg_auc = np.mean([m['auc_roc'] for m in fold_metrics])
    avg_f1  = np.mean([m['f1_score'] for m in fold_metrics])
    avg_thresh = np.mean([m['threshold'] for m in fold_metrics])
    
    print(f"\n  ── CV Average ──")
    print(f"  Accuracy:  {avg_acc:.4f}")
    print(f"  AUC-ROC:   {avg_auc:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f}")
    print(f"  Threshold: {avg_thresh:.2f}")
    print(f"  Best Fold: {best_fold_idx+1} (AUC={best_fold_auc:.4f})\n")
    
    # ============================================================
    # PHASE 2: RETRAIN ON ALL DATA (FINAL MODEL)
    # ============================================================
    print("─" * 70)
    print(f"  PHASE 2: RETRAIN ON ALL {len(y_local)} PATIENTS (FINAL MODEL)")
    print("─" * 70 + "\n")
    
    final_model = train_one_fold(X_local, y_local, epochs=LOCAL_EPOCHS, lr=LOCAL_LR)
    
    # Evaluate final model on all data
    final_metrics = evaluate_fold(final_model, X_local, y_local)
    
    print(f"  Final Model (all data):")
    print(f"    Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"    AUC-ROC:   {final_metrics['auc_roc']:.4f}")
    print(f"    F1-Score:  {final_metrics['f1_score']:.4f}")
    print(f"    Threshold: {final_metrics['threshold']:.2f}\n")
    
    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    
    # Save model
    model_path = os.path.join(HOSPITAL_DIR, 'local_model.pth')
    torch.save(final_model.state_dict(), model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save CV results
    cv_results = {
        'n_folds': N_FOLDS,
        'fold_metrics': [{k: float(v) for k, v in m.items()} for m in fold_metrics],
        'cv_average': {
            'accuracy': float(avg_acc),
            'auc_roc': float(avg_auc),
            'f1_score': float(avg_f1),
            'threshold': float(avg_thresh)
        },
        'best_fold': best_fold_idx + 1,
        'final_model_metrics': {k: float(v) for k, v in final_metrics.items()}
    }
    
    cv_path = os.path.join(HOSPITAL_DIR, 'cv_results.json')
    with open(cv_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"  ✓ CV results saved: {cv_path}")
    
    # Save training report
    report_path = os.path.join(HOSPITAL_DIR, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"HOSPITAL {HOSPITAL_NAME} TRAINING REPORT ({N_FOLDS}-FOLD CV)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset:\n")
        f.write(f"  Patients: {len(y_local)}  |  Survived: {int(len(y_local)-y_local.sum())}  |  Died: {int(y_local.sum())}\n\n")
        
        f.write(f"Training Config:\n")
        f.write(f"  Epochs: {LOCAL_EPOCHS}  |  LR: {LOCAL_LR}  |  Batch: {BATCH_SIZE}\n")
        f.write(f"  Loss: BCEWithLogitsLoss (weighted)  |  Optimizer: Adam\n\n")
        
        f.write(f"Cross-Validation Results:\n")
        f.write(f"  {'Fold':<6} {'Acc':<10} {'AUC':<10} {'F1':<10} {'Thresh':<10}\n")
        f.write(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
        for i, m in enumerate(fold_metrics):
            f.write(f"  {i+1:<6} {m['accuracy']:<10.4f} {m['auc_roc']:<10.4f} {m['f1_score']:<10.4f} {m['threshold']:<10.2f}\n")
        f.write(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
        f.write(f"  {'AVG':<6} {avg_acc:<10.4f} {avg_auc:<10.4f} {avg_f1:<10.4f} {avg_thresh:<10.2f}\n\n")
        
        f.write(f"Final Model (retrained on all {len(y_local)} patients):\n")
        f.write(f"  Accuracy: {final_metrics['accuracy']:.4f}\n")
        f.write(f"  AUC-ROC:  {final_metrics['auc_roc']:.4f}\n")
        f.write(f"  F1-Score: {final_metrics['f1_score']:.4f}\n")
    
    print(f"  ✓ Report saved: {report_path}")
    
    print(f"\n{'=' * 70}")
    print(f"  HOSPITAL {HOSPITAL_NAME} COMPLETE")
    print(f"{'=' * 70}\n")
