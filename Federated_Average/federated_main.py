"""
===============================================================================
 FEDERATED LEARNING WITH CKKS-RNS ENCRYPTED AGGREGATION
===============================================================================

 This script implements the COMPLETE pipeline:

 ┌──────────────────────────────────────────────────────────────┐
 │  DATA SPLIT STRATEGY                                        │
 │                                                              │
 │  123 patients ──────► Stratified Split by mortality label    │
 │  (87 survived, 36 died)                                     │
 │                                                              │
 │  Hospital A:  41 patients  (29 survived, 12 died)            │
 │  Hospital B:  41 patients  (29 survived, 12 died)            │
 │  Hospital C:  41 patients  (29 survived, 12 died)            │
 │                                                              │
 │  WHY stratified? Each hospital gets ~29% mortality rate,     │
 │  matching the global ratio. Otherwise one hospital might     │
 │  get all "died" cases → biased local model.                  │
 └──────────────────────────────────────────────────────────────┘

 ┌──────────────────────────────────────────────────────────────┐
 │  FEDERATED ROUND (repeated R times)                         │
 │                                                              │
 │  1. Server sends  w_global  to all hospitals                 │
 │  2. Each hospital:                                           │
 │       w_local = LocalTrain(w_global, local_data, E epochs)   │
 │       ct_local = CKKS_Encrypt(w_local)   ← encrypted        │
 │       Send ct_local to server                                │
 │  3. Server (BLIND aggregation):                              │
 │       ct_sum = ct_A ⊕ ct_B ⊕ ct_C       ← homomorphic add  │
 │       ct_avg = (1/3) ⊗ ct_sum            ← homomorphic mul  │
 │  4. Each hospital decrypts:                                  │
 │       w_global_new = Decrypt(ct_avg)                         │
 │  5. Repeat from step 1 with w_global_new                     │
 └──────────────────────────────────────────────────────────────┘

===============================================================================
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# Add parent paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'Model_Training'))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'Encryption'))

from LSTM import MortalityLSTM, load_and_reshape, FEATURE_COLS, TARGET_COL, MAX_DAYS, NUM_FEATURES
from ecyption_Tenseal_RNS import (
    create_ckks_context,
    encrypt_weights,
    decrypt_weights,
    aggregate_encrypted,
    flatten_model_weights,
    reconstruct_model_weights
)

# ============================================================
# CONFIGURATION
# ============================================================
NUM_HOSPITALS = 3                # K = number of federated participants
FEDERATED_ROUNDS = 10           # R = communication rounds
LOCAL_EPOCHS = 30               # E = local training epochs per round
LOCAL_LR = 0.002                # Local learning rate
BATCH_SIZE = 16                 # Smaller batch for smaller local data
RANDOM_SEED = 42

DATA_PATH = os.path.join(BASE_DIR, '..', 'Dataset', 'processed', 'mimic', 'mimic_ppwindowed_dataset.csv')
PTH_PATH = os.path.join(BASE_DIR, '..', 'models', 'lstm_baseline.pth')

# ============================================================
# STEP 1: SPLIT DATA INTO K HOSPITAL SUBSETS
# ============================================================
def split_data_for_hospitals(X, y, num_hospitals=NUM_HOSPITALS, seed=RANDOM_SEED):
    """
    Splits 123 patients into K non-overlapping hospital subsets.
    
    Strategy: STRATIFIED split — each hospital gets the same mortality ratio.
    
    With 123 patients and K=3:
      Hospital A: 41 patients (~29 survived, ~12 died)
      Hospital B: 41 patients (~29 survived, ~12 died)
      Hospital C: 41 patients (~29 survived, ~12 died)
    
    Stratification ensures:
      - No hospital gets only "survived" patients
      - Each hospital has a representative sample
      - Local models don't suffer from extreme class imbalance
    
    Args:
        X: (123, 8, 8) patient sequences
        y: (123,) mortality labels
        num_hospitals: K
        seed: random seed for reproducibility
    
    Returns:
        hospital_data: list of K dicts, each with {'X': ..., 'y': ...}
    """
    np.random.seed(seed)
    
    n_patients = len(y)
    indices = np.arange(n_patients)
    
    # Stratified split: preserve mortality ratio in each hospital
    # We do this iteratively: split into [hospital_A | rest], then [hospital_B | hospital_C]
    hospital_data = []
    remaining_indices = indices.copy()
    remaining_y = y.copy()
    
    for i in range(num_hospitals - 1):
        # How many patients go to this hospital
        hospitals_left = num_hospitals - i
        n_this_hospital = len(remaining_indices) // hospitals_left
        
        # Stratified split: this_hospital vs rest
        # test_size = fraction for "rest"
        frac_this = n_this_hospital / len(remaining_indices)
        
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_this_hospital, random_state=seed + i)
        this_idx, rest_idx = next(sss.split(remaining_indices, remaining_y))
        
        # Get actual patient indices
        actual_this = remaining_indices[this_idx]
        actual_rest = remaining_indices[rest_idx]
        
        hospital_data.append({
            'X': X[actual_this],
            'y': y[actual_this]
        })
        
        # Update remaining
        remaining_indices = actual_rest
        remaining_y = y[actual_rest]
    
    # Last hospital gets all remaining
    hospital_data.append({
        'X': X[remaining_indices],
        'y': y[remaining_indices]
    })
    
    # Print distribution
    print("\n--- Hospital Data Split ---")
    for i, data in enumerate(hospital_data):
        n = len(data['y'])
        died = int(data['y'].sum())
        survived = n - died
        print(f"  Hospital {chr(65+i)}: {n:3d} patients  "
              f"(Survived={survived}, Died={died}, "
              f"Mortality={died/n*100:.1f}%)")
    
    total = sum(len(d['y']) for d in hospital_data)
    print(f"  Total: {total} patients (no overlap)")
    
    return hospital_data


# ============================================================
# STEP 2: LOCAL TRAINING (Each hospital trains independently)
# ============================================================
def local_train(model, X_local, y_local, epochs=LOCAL_EPOCHS, lr=LOCAL_LR):
    """
    Hospital trains its local model for E epochs.
    
    Input:  global model weights (from previous round's aggregate)
    Output: locally updated weights after training on THIS hospital's data
    
    The key insight: each hospital ONLY sees its own patients.
    It cannot see other hospitals' data. Only the encrypted weights
    are shared with the server.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Create local DataLoader
    X_t = torch.tensor(X_local, dtype=torch.float32)
    y_t = torch.tensor(y_local, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Weighted loss for class imbalance
    num_pos = y_local.sum()
    num_neg = len(y_local) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        total_loss = epoch_loss
    
    return model, total_loss


# ============================================================
# STEP 3: EVALUATION
# ============================================================
def evaluate_global_model(model, X, y, label=""):
    """
    Evaluates the global model on ALL data (combined from all hospitals).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_t).cpu().numpy().flatten()
        y_proba = 1 / (1 + np.exp(-logits))
        y_true = y.astype(int)
    
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
# STEP 4: FEDERATED LEARNING WITH CKKS-RNS ENCRYPTION
# ============================================================
def federated_learning_encrypted(hospital_data, X_all, y_all, hospital_folders=None):
    """
    Complete Federated Learning pipeline with CKKS-RNS encrypted aggregation.
    
    Saves each hospital's locally trained weights to their respective folder.
    
    ┌──────────────────────────────────────────────────────────────┐
    │  For each round r = 1, 2, ..., R:                           │
    │                                                              │
    │  1. Distribute w_global to all hospitals                     │
    │  2. Hospital k trains:                                       │
    │       w_k = LocalTrain(w_global, D_k, E local epochs)       │
    │       Save w_k to Hospital_k/weights_round_r.pth            │
    │  3. Hospital k encrypts:                                     │
    │       ct_k = CKKS.Encrypt(w_k)                              │
    │  4. Server aggregates (BLIND — no plaintext access):         │
    │       ct_avg = (1/K) ⊗ (ct_1 ⊕ ct_2 ⊕ ... ⊕ ct_K)         │
    │  5. Hospital decrypts:                                       │
    │       w_global = CKKS.Decrypt(ct_avg)                        │
    └──────────────────────────────────────────────────────────────┘
    
    Args:
        hospital_data: list of dicts with 'X' and 'y'
        X_all: all training data (for evaluation)
        y_all: all labels (for evaluation)
        hospital_folders: optional list of folder paths for each hospital
    """
    print("=" * 70)
    print("  FEDERATED LEARNING WITH CKKS-RNS ENCRYPTED AGGREGATION")
    print("=" * 70)
    print(f"  Hospitals:        {NUM_HOSPITALS}")
    print(f"  Fed Rounds (R):   {FEDERATED_ROUNDS}")
    print(f"  Local Epochs (E): {LOCAL_EPOCHS}")
    print(f"  Total Params:     5,921 per model")
    print("=" * 70)
    
    # ---- Create CKKS Context (shared keys for demo) ----
    print("\n--- Setting up CKKS-RNS Encryption ---")
    context = create_ckks_context()
    
    # ---- Initialize global model ----
    # Option A: From pretrained .pth
    # Option B: Random init (both hospitals start from same point)
    global_model = MortalityLSTM()
    
    if os.path.exists(PTH_PATH):
        state_dict = torch.load(PTH_PATH, map_location='cpu', weights_only=True)
        global_model.load_state_dict(state_dict)
        print(f"\n  Initialized from pretrained: lstm_baseline.pth")
    else:
        print(f"\n  Initialized with random weights")
    
    # ---- Track metrics per round ----
    round_metrics = []
    
    # =======================================
    # FEDERATED ROUNDS
    # =======================================
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        print(f"\n{'─' * 70}")
        print(f"  ROUND {round_num}/{FEDERATED_ROUNDS}")
        print(f"{'─' * 70}")
        
        all_encrypted_chunks = []
        weight_shapes = None
        round_enc_time = 0
        round_train_losses = []
        
        # ---- Step 1 & 2: Each hospital trains locally ----
        for h_idx, h_data in enumerate(hospital_data):
            hospital_name = chr(65 + h_idx)
            
            # Clone global model → hospital gets a fresh copy
            local_model = MortalityLSTM()
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            
            # Train on this hospital's LOCAL data only
            local_model, loss = local_train(
                local_model, h_data['X'], h_data['y'],
                epochs=LOCAL_EPOCHS, lr=LOCAL_LR
            )
            round_train_losses.append(loss)
            
            # ---- Save locally trained weights ----
            if hospital_folders is not None and h_idx < len(hospital_folders):
                weights_path = os.path.join(
                    hospital_folders[h_idx],
                    f"weights_round_{round_num:02d}.pth"
                )
                torch.save(local_model.state_dict(), weights_path)
                
                # Save round summary
                summary_path = os.path.join(
                    hospital_folders[h_idx],
                    f"training_summary_round_{round_num:02d}.txt"
                )
                with open(summary_path, 'w') as f:
                    f.write(f"Hospital {hospital_name} - Round {round_num} Training Summary\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Local Training Loss: {loss:.6f}\n")
                    f.write(f"Local Epochs: {LOCAL_EPOCHS}\n")
                    f.write(f"Learning Rate: {LOCAL_LR}\n")
                    f.write(f"Batch Size: {BATCH_SIZE}\n")
                    f.write(f"Local Patients: {len(h_data['y'])}\n")
                    f.write(f"Model Parameters: 5,921\n")
            
            # ---- Step 3: Encrypt local weights ----
            enc_chunks, shapes, enc_time = encrypt_weights(local_model, context)
            all_encrypted_chunks.append(enc_chunks)
            weight_shapes = shapes
            round_enc_time += enc_time
            
            print(f"    Hospital {hospital_name}: trained on {len(h_data['y'])} patients, "
                  f"loss={loss:.4f}, encrypted in {enc_time:.4f}s", end="")
            if hospital_folders is not None:
                print(f", weights saved ✓")
            else:
                print()
        
        # ---- Step 4: Server aggregates ENCRYPTED weights (BLIND) ----
        print(f"\n    Server: aggregating {NUM_HOSPITALS} encrypted models...")
        agg_chunks, agg_time = aggregate_encrypted(
            all_encrypted_chunks, NUM_HOSPITALS
        )
        
        # ---- Step 5: Decrypt aggregated weights → new global model ----
        agg_state_dict, dec_time = decrypt_weights(agg_chunks, weight_shapes)
        
        # Load aggregated weights into global model
        global_model.load_state_dict(agg_state_dict)
        
        # ---- Evaluate global model ----
        metrics = evaluate_global_model(global_model, X_all, y_all, label=f"Round {round_num}")
        metrics['round'] = round_num
        metrics['enc_time'] = round_enc_time
        metrics['agg_time'] = agg_time
        metrics['dec_time'] = dec_time
        metrics['avg_loss'] = np.mean(round_train_losses)
        round_metrics.append(metrics)
        
        print(f"\n    Global Model Metrics:")
        print(f"      Acc={metrics['accuracy']:.4f}  "
              f"AUC={metrics['auc_roc']:.4f}  "
              f"F1={metrics['f1_score']:.4f}  "
              f"Loss={metrics['avg_loss']:.4f}")
        print(f"      Crypto overhead: enc={round_enc_time:.4f}s "
              f"agg={agg_time:.4f}s dec={dec_time:.4f}s "
              f"total={round_enc_time+agg_time+dec_time:.4f}s")
    
    return global_model, round_metrics


# ============================================================
# STEP 5: COMPARISON — PLAINTEXT vs ENCRYPTED FEDERATED
# ============================================================
def plaintext_federated(hospital_data, X_all, y_all):
    """
    Same FedAvg but WITHOUT encryption (plaintext).
    Used to verify encrypted version produces identical results.
    """
    print("\n" + "=" * 70)
    print("  PLAINTEXT FEDERATED LEARNING (No Encryption — Baseline)")
    print("=" * 70)
    
    global_model = MortalityLSTM()
    
    if os.path.exists(PTH_PATH):
        state_dict = torch.load(PTH_PATH, map_location='cpu', weights_only=True)
        global_model.load_state_dict(state_dict)
    
    round_metrics = []
    
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        local_state_dicts = []
        
        for h_idx, h_data in enumerate(hospital_data):
            local_model = MortalityLSTM()
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model, _ = local_train(local_model, h_data['X'], h_data['y'])
            local_state_dicts.append(local_model.state_dict())
        
        # Plaintext FedAvg: simple arithmetic average
        avg_state = {}
        for key in local_state_dicts[0]:
            avg_state[key] = sum(sd[key] for sd in local_state_dicts) / NUM_HOSPITALS
        
        global_model.load_state_dict(avg_state)
        metrics = evaluate_global_model(global_model, X_all, y_all)
        metrics['round'] = round_num
        round_metrics.append(metrics)
        
        if round_num % 2 == 0 or round_num == 1:
            print(f"  Round {round_num}: Acc={metrics['accuracy']:.4f}  "
                  f"AUC={metrics['auc_roc']:.4f}  F1={metrics['f1_score']:.4f}")
    
    return global_model, round_metrics


# ============================================================
# STEP 6: SAVE HOSPITAL DATASETS TO DISK
# ============================================================
def save_hospital_datasets(hospital_data, output_dir=None):
    """
    Saves each hospital's dataset (X and y) to separate folders.
    
    Structure:
      Hospital_A/
        X.npy          (41, 8, 8) - sequences
        y.npy          (41,)      - labels
        metadata.txt   - info
      
      Hospital_B/ ... Hospital_C/ ...
    
    Args:
        hospital_data: list of K dicts with 'X' and 'y' keys
        output_dir: where to save folders (default: current dir)
    
    Returns:
        List of saved folder paths
    """
    if output_dir is None:
        output_dir = BASE_DIR
    
    saved_paths = []
    
    for h_idx, h_data in enumerate(hospital_data):
        hospital_name = chr(65 + h_idx)  # A, B, C
        hospital_folder = os.path.join(output_dir, f"Hospital_{hospital_name}")
        
        # Create folder
        os.makedirs(hospital_folder, exist_ok=True)
        
        # Save X and y
        X = h_data['X']
        y = h_data['y']
        
        X_path = os.path.join(hospital_folder, "X.npy")
        y_path = os.path.join(hospital_folder, "y.npy")
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        # Save metadata
        n_patients = len(y)
        n_died = int(y.sum())
        n_survived = n_patients - n_died
        mortality_rate = (n_died / n_patients) * 100 if n_patients > 0 else 0
        
        metadata_path = os.path.join(hospital_folder, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Hospital {hospital_name} Dataset\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of Patients: {n_patients}\n")
            f.write(f"Survived: {n_survived}\n")
            f.write(f"Died: {n_died}\n")
            f.write(f"Mortality Rate: {mortality_rate:.1f}%\n\n")
            f.write(f"X shape: {X.shape}  (patients, max_days, features)\n")
            f.write(f"y shape: {y.shape}  (patients,)\n\n")
            f.write(f"Files:\n")
            f.write(f"  - X.npy: Patient sequences (41, 8, 8 or similar)\n")
            f.write(f"  - y.npy: Mortality labels (0 or 1)\n")
        
        saved_paths.append(hospital_folder)
        
        print(f"  Hospital {hospital_name}: saved to {hospital_folder}")
        print(f"    - X.npy ({X.shape})")
        print(f"    - y.npy ({y.shape})")
        print(f"    - metadata.txt")
    
    print(f"\n  ✓ All hospital datasets saved\n")
    return saved_paths


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    # ---- Load all data ----
    print("--- Loading MIMIC-III Data ---")
    X_all, y_all, patient_ids = load_and_reshape(DATA_PATH)
    
    # ---- Split into hospital subsets ----
    hospital_data = split_data_for_hospitals(X_all, y_all, NUM_HOSPITALS)
    
    # ---- Save hospital datasets to disk ----
    print("\n--- Saving Hospital Datasets ---")
    hospital_folders = save_hospital_datasets(hospital_data, output_dir=BASE_DIR)
    
    # ---- Run Encrypted Federated Learning ----
    enc_model, enc_metrics = federated_learning_encrypted(hospital_data, X_all, y_all, 
                                                         hospital_folders=hospital_folders)
    
    # ---- Run Plaintext Federated Learning (for comparison) ----
    pt_model, pt_metrics = plaintext_federated(hospital_data, X_all, y_all)
    
    # ---- Final Comparison ----
    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON: ENCRYPTED vs PLAINTEXT FEDERATED LEARNING")
    print("=" * 70)
    
    print(f"\n  {'Round':<8} {'Enc Acc':<10} {'PT Acc':<10} {'Enc AUC':<10} "
          f"{'PT AUC':<10} {'Enc F1':<10} {'PT F1':<10} {'Diff':<10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    
    for em, pm in zip(enc_metrics, pt_metrics):
        r = em['round']
        diff_acc = abs(em['accuracy'] - pm['accuracy'])
        print(f"  {r:<8} {em['accuracy']:<10.4f} {pm['accuracy']:<10.4f} "
              f"{em['auc_roc']:<10.4f} {pm['auc_roc']:<10.4f} "
              f"{em['f1_score']:<10.4f} {pm['f1_score']:<10.4f} "
              f"{diff_acc:<10.2e}")
    
    # Final round results
    enc_final = enc_metrics[-1]
    pt_final = pt_metrics[-1]
    
    print(f"\n  FINAL ENCRYPTED:  Acc={enc_final['accuracy']:.4f}  "
          f"AUC={enc_final['auc_roc']:.4f}  F1={enc_final['f1_score']:.4f}")
    print(f"  FINAL PLAINTEXT:  Acc={pt_final['accuracy']:.4f}  "
          f"AUC={pt_final['auc_roc']:.4f}  F1={pt_final['f1_score']:.4f}")
    
    acc_diff = abs(enc_final['accuracy'] - pt_final['accuracy'])
    print(f"\n  Accuracy Difference: {acc_diff:.2e}")
    
    if acc_diff < 0.01:
        print("  ✓ ENCRYPTION INTRODUCES NO MEANINGFUL PERFORMANCE LOSS")
    else:
        print("  ⚠ Performance differs — investigate noise accumulation")
    
    # Timing summary
    total_enc = sum(m['enc_time'] for m in enc_metrics)
    total_agg = sum(m['agg_time'] for m in enc_metrics)
    total_dec = sum(m['dec_time'] for m in enc_metrics)
    total_crypto = total_enc + total_agg + total_dec
    
    print(f"\n  Total Crypto Overhead ({FEDERATED_ROUNDS} rounds):")
    print(f"    Encryption: {total_enc:.4f}s")
    print(f"    Aggregation: {total_agg:.4f}s")
    print(f"    Decryption: {total_dec:.4f}s")
    print(f"    TOTAL: {total_crypto:.4f}s")
    print("=" * 70)
    
    # Save federated model
    save_path = os.path.join(BASE_DIR, '..', 'models', 'federated_encrypted_model.pth')
    torch.save(enc_model.state_dict(), save_path)
    print(f"\n  Federated model saved to: {save_path}")
