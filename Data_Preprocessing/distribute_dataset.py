"""
===============================================================================
 DATASET DISTRIBUTION TO 3 HOSPITALS
===============================================================================

This script:
1. Loads the MIMIC-III preprocessed CSV
2. Reshapes into (123, 8, 8) for LSTM
3. Splits into 3 hospitals using STRATIFIED sampling
4. Saves each hospital's dataset as X.npy and y.npy

Output:
  Hospital_A/
    ├── X.npy (41, 8, 8)
    ├── y.npy (41,)
    └── metadata.txt
  Hospital_B/ (same)
  Hospital_C/ (same)

===============================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ============================================================
# CONFIGURATION
# ============================================================
# Get project root (go up from Data_Preprocessing to parent)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level to project root

CSV_PATH = os.path.join(PROJECT_ROOT, "Dataset", "processed", "mimic", "mimic_ppwindowed_dataset.csv")

# LSTM parameters (must match LSTM.py)
MAX_DAYS = 8
FEATURE_COLS = ['age', 'gender', 'HeartRate', 'SysBP', 'RespRate', 'Temp', 'SpO2', 'Glucose']
TARGET_COL = 'hospital_expire_flag'
NUM_FEATURES = len(FEATURE_COLS)

# Hospital split parameters
NUM_HOSPITALS = 3
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Model_Training")
RANDOM_SEED = 42

# ============================================================
# STEP 1: LOAD & RESHAPE CSV INTO 3D TENSORS
# ============================================================
def load_and_reshape_data(csv_path):
    """
    Loads CSV and reshapes into LSTM-ready tensors.
    
    Input:  (437 rows, 11 cols) CSV
    Output: X = (123, 8, 8), y = (123,)
    """
    print("=" * 70)
    print("  LOADING & RESHAPING DATA")
    print("=" * 70 + "\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded CSV: {df.shape[0]} rows, {df['hadm_id'].nunique()} unique patients\n")
    
    # Group by patient
    grouped = df.groupby('hadm_id')
    
    sequences = []
    labels = []
    patient_ids = []
    
    for hadm_id, group in grouped:
        # Sort by day
        group = group.sort_values('day_num')
        
        # Extract features (8 features × num_days)
        feat_matrix = group[FEATURE_COLS].values
        
        # Pad to MAX_DAYS=8
        num_days = feat_matrix.shape[0]
        if num_days < MAX_DAYS:
            padding = np.zeros((MAX_DAYS - num_days, NUM_FEATURES))
            feat_matrix = np.vstack([feat_matrix, padding])
        else:
            feat_matrix = feat_matrix[:MAX_DAYS]
        
        sequences.append(feat_matrix)
        labels.append(group[TARGET_COL].iloc[0])
        patient_ids.append(hadm_id)
    
    X = np.array(sequences, dtype=np.float32)  # (123, 8, 8)
    y = np.array(labels, dtype=np.float32)      # (123,)
    
    print(f"✓ Reshaped data:")
    print(f"  X shape: {X.shape} (patients, max_days, features)")
    print(f"  y shape: {y.shape} (patients,)")
    print(f"  Survived: {int((y==0).sum())}")
    print(f"  Died: {int((y==1).sum())}")
    print(f"  Mortality Rate: {(y.sum() / len(y)) * 100:.1f}%\n")
    
    return X, y, patient_ids


# ============================================================
# STEP 2: STRATIFIED SPLIT INTO 3 HOSPITALS
# ============================================================
def split_into_hospitals(X, y, num_hospitals=NUM_HOSPITALS, seed=RANDOM_SEED):
    """
    Splits 123 patients into K hospital subsets using STRATIFIED sampling.
    
    Each hospital gets the same mortality rate (~29%).
    
    Returns:
        List of K dicts: [{'X': ..., 'y': ...}, ...]
    """
    print("=" * 70)
    print("  SPLIT INTO HOSPITALS (STRATIFIED)")
    print("=" * 70 + "\n")
    
    np.random.seed(seed)
    
    n_patients = len(y)
    indices = np.arange(n_patients)
    
    hospital_data = []
    remaining_indices = indices.copy()
    remaining_y = y.copy()
    
    # Iteratively split: Hospital_A from rest, then Hospital_B from remaining
    for i in range(num_hospitals - 1):
        hospitals_left = num_hospitals - i
        n_this_hospital = len(remaining_indices) // hospitals_left
        
        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_this_hospital, random_state=seed + i)
        this_idx, rest_idx = next(sss.split(remaining_indices, remaining_y))
        
        actual_this = remaining_indices[this_idx]
        actual_rest = remaining_indices[rest_idx]
        
        hospital_data.append({
            'X': X[actual_this],
            'y': y[actual_this]
        })
        
        remaining_indices = actual_rest
        remaining_y = y[actual_rest]
    
    # Last hospital gets remaining
    hospital_data.append({
        'X': X[remaining_indices],
        'y': y[remaining_indices]
    })
    
    # Print distribution
    print(f"Hospital Distribution:\n")
    for i, data in enumerate(hospital_data):
        hospital_name = chr(65 + i)  # A, B, C
        n = len(data['y'])
        died = int(data['y'].sum())
        survived = n - died
        mortality_rate = (died / n * 100) if n > 0 else 0
        
        print(f"  Hospital {hospital_name}: {n:3d} patients")
        print(f"    ├─ Survived: {survived}")
        print(f"    ├─ Died: {died}")
        print(f"    └─ Mortality Rate: {mortality_rate:.1f}%\n")
    
    total = sum(len(d['y']) for d in hospital_data)
    print(f"  Total: {total} patients (no overlap)\n")
    
    return hospital_data


# ============================================================
# STEP 3: SAVE HOSPITAL DATASETS
# ============================================================
def save_hospital_datasets(hospital_data, output_dir):
    """
    Saves each hospital's X and y as .npy files.
    
    Creates folders:
      Hospital_A/
        ├── X.npy (41, 8, 8)
        ├── y.npy (41,)
        └── metadata.txt
    """
    print("=" * 70)
    print("  SAVING HOSPITAL DATASETS")
    print("=" * 70 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    hospital_dirs = []
    
    for h_idx, h_data in enumerate(hospital_data):
        hospital_name = chr(65 + h_idx)  # A, B, C
        hospital_dir = os.path.join(output_dir, f"Hospital_{hospital_name}")
        
        os.makedirs(hospital_dir, exist_ok=True)
        hospital_dirs.append(hospital_dir)
        
        # Save X and y
        X = h_data['X']
        y = h_data['y']
        
        X_path = os.path.join(hospital_dir, 'X.npy')
        y_path = os.path.join(hospital_dir, 'y.npy')
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        # Save metadata
        n = len(y)
        died = int(y.sum())
        survived = n - died
        mortality_rate = (died / n * 100) if n > 0 else 0
        
        metadata_path = os.path.join(hospital_dir, 'metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"HOSPITAL {hospital_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset Information:\n")
            f.write(f"  Total Patients: {n}\n")
            f.write(f"  Survived: {survived}\n")
            f.write(f"  Died: {died}\n")
            f.write(f"  Mortality Rate: {mortality_rate:.1f}%\n\n")
            f.write(f"Data Shape:\n")
            f.write(f"  X: {X.shape} (patients, max_days, features)\n")
            f.write(f"  y: {y.shape} (patients,)\n\n")
            f.write(f"Features (8):\n")
            f.write(f"  1. age\n")
            f.write(f"  2. gender\n")
            f.write(f"  3. HeartRate\n")
            f.write(f"  4. SysBP\n")
            f.write(f"  5. RespRate\n")
            f.write(f"  6. Temp\n")
            f.write(f"  7. SpO2\n")
            f.write(f"  8. Glucose\n\n")
            f.write(f"Target:\n")
            f.write(f"  hospital_expire_flag: 0=Survived, 1=Died\n")
        
        print(f"✓ Hospital {hospital_name}: saved to {hospital_dir}")
        print(f"    ├─ X.npy ({X.shape})")
        print(f"    ├─ y.npy ({y.shape})")
        print(f"    └─ metadata.txt\n")
    
    return hospital_dirs


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    
    print("\n")
    print(f"DEBUG: CSV path = {CSV_PATH}")
    print(f"DEBUG: CSV exists = {os.path.exists(CSV_PATH)}")
    print(f"DEBUG: Output dir = {OUTPUT_DIR}\n")
    
    # Step 1: Load & reshape
    X, y, patient_ids = load_and_reshape_data(CSV_PATH)
    
    # Step 2: Split into hospitals
    hospital_data = split_into_hospitals(X, y, num_hospitals=NUM_HOSPITALS)
    
    # Step 3: Save
    hospital_dirs = save_hospital_datasets(hospital_data, OUTPUT_DIR)
    
    print("=" * 70)
    print("  DISTRIBUTION COMPLETE")
    print("=" * 70)
    print(f"\n✓ All datasets saved to: {OUTPUT_DIR}\n")
    print("Hospital folders ready for training:\n")
    for h_dir in hospital_dirs:
        print(f"  - {h_dir}\n")
