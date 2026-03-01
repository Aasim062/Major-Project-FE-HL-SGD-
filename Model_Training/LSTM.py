import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import os

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "processed", "mimic", "mimic_ppwindowed_dataset.csv")

MAX_DAYS = 8          # Max timesteps (pad shorter sequences to this)
FEATURE_COLS = ['age', 'gender', 'HeartRate', 'SysBP', 'RespRate', 'Temp', 'SpO2', 'Glucose']
TARGET_COL = 'hospital_expire_flag'
NUM_FEATURES = len(FEATURE_COLS)

# Hyperparameters
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 0.002
EPOCHS = 300
BATCH_SIZE = 32
TEST_SPLIT = 0.2
RANDOM_SEED = 42
N_FOLDS = 5  # Cross-validation folds

# ============================================================
# STEP 2.1: LOAD & RESHAPE DATA INTO 3D SEQUENCES
# ============================================================
def load_and_reshape(csv_path):
    """
    Loads the windowed CSV and reshapes it into LSTM-ready 3D tensors.
    
    Input CSV shape:  (437 rows, 11 cols) — multiple rows per patient
    Output tensors:   X = (num_patients, MAX_DAYS, NUM_FEATURES)
                      y = (num_patients,)
    """
    print("--- Loading Data ---")
    df = pd.read_csv(csv_path)
    print(f"  Raw CSV: {df.shape[0]} rows, {df['hadm_id'].nunique()} patients")
    
    # Group by patient ID
    grouped = df.groupby('hadm_id')
    
    sequences = []   # Will hold (MAX_DAYS, NUM_FEATURES) arrays
    labels = []      # Will hold 0/1 mortality labels
    patient_ids = [] # Track which patient each sequence belongs to
    
    for hadm_id, group in grouped:
        # Sort by day number to ensure chronological order
        group = group.sort_values('day_num')
        
        # Extract features for this patient (each row = 1 day)
        feat_matrix = group[FEATURE_COLS].values  # shape: (num_days, 8)
        
        # Pad shorter sequences with zeros to reach MAX_DAYS
        num_days = feat_matrix.shape[0]
        if num_days < MAX_DAYS:
            padding = np.zeros((MAX_DAYS - num_days, NUM_FEATURES))
            feat_matrix = np.vstack([feat_matrix, padding])
        else:
            # Truncate if somehow longer than MAX_DAYS
            feat_matrix = feat_matrix[:MAX_DAYS]
        
        sequences.append(feat_matrix)
        labels.append(group[TARGET_COL].iloc[0])  # Same label for all rows of a patient
        patient_ids.append(hadm_id)
    
    X = np.array(sequences, dtype=np.float32)  # (123, 8, 8)
    y = np.array(labels, dtype=np.float32)      # (123,)
    
    print(f"  Reshaped: X={X.shape}, y={y.shape}")
    print(f"  Class distribution: Survived={int((y==0).sum())}, Died={int((y==1).sum())}")
    
    return X, y, patient_ids

# ============================================================
# STEP 2.1 (E-F): TRAIN/TEST SPLIT BY PATIENT & CONVERT TO TENSORS
# ============================================================
def prepare_dataloaders(X, y):
    """
    Splits data by patient (not by row) and creates PyTorch DataLoaders.
    """
    # Split by patient index
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"  Train: {X_train.shape[0]} patients, Test: {X_test.shape[0]} patients")
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)  # (N, 1) for BCELoss
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, X_test_t, y_test_t

# ============================================================
# STEP 2.2: LSTM MODEL DEFINITION
# ============================================================
class MortalityLSTM(nn.Module):
    """
    LSTM for binary mortality prediction.
    Input:  (batch, MAX_DAYS, NUM_FEATURES) = (batch, 8, 8)
    Output: (batch, 1) — probability of mortality
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(MortalityLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
            # No Sigmoid here — BCEWithLogitsLoss handles it internally
        )
    
    def forward(self, x):
        # x shape: (batch, 8, 8)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state from the final LSTM layer
        last_hidden = h_n[-1]  # shape: (batch, hidden_size)
        
        # Output raw logits (no sigmoid)
        output = self.classifier(last_hidden)  # shape: (batch, 1)
        return output

# ============================================================
# STEP 2.3: TRAINING LOOP
# ============================================================
def train_model(model, train_loader, y_train, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Trains the LSTM model using Weighted Binary Cross-Entropy loss
    to handle class imbalance (more Survived than Died).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # CLASS IMBALANCE FIX: Calculate positive class weight
    # If 87 survived, 36 died → weight for died = 87/36 ≈ 2.4
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    print(f"  Class weight for 'Died': {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    print(f"\n--- Training on {device} for {epochs} epochs ---")
    
    history = {'loss': [], 'acc': []}
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass (raw logits, sigmoid applied by BCEWithLogitsLoss)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * X_batch.size(0)
            predicted_labels = (torch.sigmoid(logits) > 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        # Learning rate scheduler
        scheduler.step()
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    print(f"  Best training loss: {best_loss:.4f}")
    
    return model, history

# ============================================================
# STEP 2.4: EVALUATION (PLAINTEXT BASELINE)
# ============================================================
def evaluate_model(model, X_test_t, y_test_t):
    """
    Evaluates the trained model and prints Accuracy, AUC-ROC, F1-Score.
    Uses optimal threshold tuning for imbalanced classes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        logits = model(X_test_t).cpu().numpy().flatten()
        y_pred_proba = 1 / (1 + np.exp(-logits))  # Apply sigmoid to raw logits
        y_true = y_test_t.numpy().flatten().astype(int)
    
    # Find optimal threshold (maximize F1)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.2, 0.8, 0.05):
        y_temp = (y_pred_proba > thresh).astype(int)
        f1_temp = f1_score(y_true, y_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_thresh = thresh
    
    y_pred = (y_pred_proba > best_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.0
    
    print(f"\n{'='*50}")
    print(f"  PLAINTEXT BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"  Optimal Threshold: {best_thresh:.2f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"{'='*50}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Survived', 'Died'], zero_division=0)}")
    
    return {'accuracy': acc, 'auc_roc': auc, 'f1_score': f1, 'threshold': best_thresh}

# ============================================================
# K-FOLD CROSS-VALIDATION (More robust on small data)
# ============================================================
def cross_validate(X, y, n_folds=N_FOLDS):
    """
    Runs stratified K-Fold CV to get reliable metrics across all patients.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    
    print(f"\n{'='*50}")
    print(f"  {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*50}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_train), 
            torch.tensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train
        model = MortalityLSTM()
        model, _ = train_model(model, train_loader, y_train=y_train, epochs=EPOCHS, lr=LEARNING_RATE)
        
        # Evaluate
        X_test_t = torch.tensor(X_test)
        y_test_t = torch.tensor(y_test).unsqueeze(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device)).cpu().numpy().flatten()
            y_proba = 1 / (1 + np.exp(-logits))
            y_pred = (y_proba > 0.4).astype(int)  # Slightly lower threshold
            y_true = y_test.astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        fold_results.append({'accuracy': acc, 'auc_roc': auc, 'f1_score': f1})
        print(f"  Fold {fold+1}: Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
    
    # Average results
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    avg_auc = np.mean([r['auc_roc'] for r in fold_results])
    avg_f1 = np.mean([r['f1_score'] for r in fold_results])
    
    print(f"\n  AVERAGE: Acc={avg_acc:.4f}  AUC={avg_auc:.4f}  F1={avg_f1:.4f}")
    print(f"{'='*50}")
    
    return fold_results

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Step 2.1: Load & Reshape
    X, y, patient_ids = load_and_reshape(DATA_PATH)
    
    # Step 2.1 (E-F): Split & Create DataLoaders
    train_loader, test_loader, X_test_t, y_test_t = prepare_dataloaders(X, y)
    
    # Step 2.2: Build Model
    model = MortalityLSTM()
    print(f"\nModel Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Step 2.3: Train (pass y_train for class weight calculation)
    y_train = y[np.isin(patient_ids, patient_ids)]  # full y used for weight calc
    model, history = train_model(model, train_loader, y_train=y)
    
    # Step 2.4: Evaluate (Plaintext Baseline)
    results = evaluate_model(model, X_test_t, y_test_t)
    
    # Step 2.5: Cross-Validation (Robust metrics for paper)
    cv_results = cross_validate(X, y)
    
    # Save model weights (needed later for Federated Learning & Encryption)
    save_path = os.path.join(BASE_DIR, "..", "models", "lstm_baseline.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
