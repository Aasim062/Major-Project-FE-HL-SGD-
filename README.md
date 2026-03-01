# Privacy-Preserving Federated Learning for Dynamic Mortality Prediction using CKKS-RNS Homomorphic Encryption

> **Last Updated:** February 16, 2026

---

## Overview

This project demonstrates a secure, privacy-preserving framework for training deep learning models (LSTMs) on distributed ICU time-series data without exposing sensitive patient information.

### The Problem

- Current medical AI models rely on **centralized data**, violating privacy laws (HIPAA/GDPR).
- Existing privacy solutions (standard Federated Learning) are vulnerable to **gradient leakage attacks**.
- Most secure solutions use simple static models (Logistic Regression) that fail to capture critical **time-dependent trends** in patient deterioration.

### The Proposed Solution

A full end-to-end pipeline combining **temporal deep learning**, **homomorphic encryption**, and **federated learning**:

1. **Data Engineering** — Sliding window preprocessing to expand small datasets into rich temporal samples.
2. **LSTM Model** — Time-series mortality prediction capturing dynamic patient deterioration patterns.
3. **CKKS-RNS Encryption** — Homomorphic encryption of model weights for secure aggregation.
4. **Federated Learning** — Each stage executed as a separate script for clarity:
   - 4.1 Dataset distribution across 3 virtual hospitals
   - 4.2 Independent local training at each hospital (5-Fold CV)
   - 4.3 Encrypt local weights with CKKS-RNS *(next)*
   - 4.4 Server-side homomorphic aggregation *(pending)*
   - 4.5 Evaluation & comparison *(pending)*

---

## Progress Tracker

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Data Engineering (Sliding Window) | ✅ Complete |
| 2 | LSTM Model (Plaintext Baseline) | ✅ Complete |
| 3 | CKKS-RNS Encryption Verification | ✅ Complete |
| 4.1 | Dataset Distribution (3 Hospitals) | ✅ Complete |
| 4.2 | Local Hospital Training (5-Fold CV) | ✅ Complete |
| 4.3 | Encrypt Local Weights | ⬜ Next |
| 4.4 | Federated Aggregation (Server) | ⬜ Pending |
| 4.5 | Evaluation & Comparison | ⬜ Pending |

---

## Project Structure

```
Major-Project-FE-HL-SGD-/
|
|-- README.md                              <-- You are here
|-- .gitignore
|
|-- Dataset/
|   |-- raw/
|   |   |-- mimic/                         <-- Raw MIMIC-III Demo CSVs (gitignored)
|   |   |   |-- ADMISSIONS.csv
|   |   |   |-- PATIENTS.csv
|   |   |   |-- CHARTEVENTS.csv
|   |   |   |-- LABEVENTS.csv
|   |   |   |-- DIAGNOSES_ICD.csv
|   |   |   +-- ... (28 tables total)
|   |   +-- fda/                           <-- FDA drug-event JSON files (gitignored)
|   |       +-- drug-event-XXXX-of-0029.json/
|   +-- processed/
|       |-- mimic/                         <-- Preprocessed MIMIC outputs (gitignored)
|       |   |-- mimic_raw_combined.csv
|       |   +-- mimic_ppwindowed_dataset.csv
|       +-- fda/                           <-- Preprocessed FDA CSVs (gitignored)
|
|-- Data_Preprocessing/                    <-- Stage 1: Data Engineering
|   |-- Mimic_Raw_Combined.py              -> Static feature extraction
|   |-- mimic_pp_day_wise.py               -> 24h sliding window pipeline
|   |-- distribute_dataset.py             -> Stage 4.1: Split data into 3 hospitals
|   +-- JSON_TO_CSV.py                     -> FDA JSON -> CSV converter
|
|-- Model_Training/                        <-- Stage 2 + 4.2: LSTM Model & Local Training
|   |-- LSTM.py                            -> Model definition (MortalityLSTM class)
|   |-- Hospital_A/                        <-- Hospital A local training
|   |   |-- train_hospital_A.py            -> 5-Fold CV + retrain
|   |   |-- X.npy, y.npy                   -> Hospital A dataset (41 patients)
|   |   |-- local_model.pth                -> Trained local weights
|   |   |-- cv_results.json                -> Cross-validation metrics
|   |   +-- training_report.txt            -> Human-readable report
|   |-- Hospital_B/                        <-- Hospital B local training
|   |   |-- train_hospital_B.py
|   |   |-- X.npy, y.npy                   -> Hospital B dataset (41 patients)
|   |   |-- local_model.pth
|   |   |-- cv_results.json
|   |   +-- training_report.txt
|   +-- Hospital_C/                        <-- Hospital C local training
|       |-- train_hospital_C.py
|       |-- X.npy, y.npy                   -> Hospital C dataset (41 patients)
|       |-- local_model.pth
|       |-- cv_results.json
|       +-- training_report.txt
|
|-- Encryption/                            <-- Stage 3: CKKS-RNS Encryption
|   +-- ecyption_Tenseal_RNS.py            -> TenSEAL encrypt/decrypt/aggregate utilities
|
|-- Federated_Average/                     <-- Stage 4.4: Aggregation (pending)
|   +-- federated_main.py                  -> (to be replaced with separated scripts)
|
|-- models/                                <-- Saved model weights
|   +-- lstm_baseline.pth                  -> Plaintext baseline (5,921 params)
|
+-- logs/                                  <-- Training logs
    +-- output.txt
```

---

## Pipeline Details

### Stage 1: Data Engineering ("The Multiplier") -- Complete

- **Source:** MIMIC-III Clinical Database (Demo Subset, ~100 patients)
- **Transformation:** 24-hour sliding window preprocessing slices each patient's stay into daily time windows
- **Impact:** Expands the dataset from ~100 rows to **437 temporal training samples** across **123 admissions**
- **Features Extracted (8):** age, gender, HeartRate, SysBP, RespRate, Temp, SpO2, Glucose (normalized 0-1)
- **Scripts:** `Data_Preprocessing/Mimic_Raw_Combined.py`, `Data_Preprocessing/mimic_pp_day_wise.py`

### Stage 2: LSTM Model ("The Clinical Brain") -- Complete

- **Architecture:** LSTM(8 -> 32, 1 layer) -> Dense(32 -> 16 -> 1) -> Sigmoid
- **Total Parameters:** 5,921
- **Why LSTM:** Unlike static models, LSTMs have memory gates that connect temporal causes (e.g., drug on Day 1) to effects (e.g., organ failure on Day 4)
- **Training:** BCEWithLogitsLoss with class weighting (pos_weight=2.42), CosineAnnealing LR, gradient clipping
- **Validation:** 5-Fold Stratified Cross-Validation with optimal threshold tuning
- **Script:** `Model_Training/LSTM.py`

### Stage 3: CKKS Encryption ("The Cryptographic Shield") -- Complete

- **Scheme:** CKKS-RNS Homomorphic Encryption via [TenSEAL](https://github.com/OpenMined/TenSEAL)
- **Parameters:** poly_modulus_degree=8192, coeff_mod_bit_sizes=[60,40,40,60], scale=2^40, 128-bit security
- **Strategy:** Secure Aggregation -- hospitals encrypt trained weights, server aggregates blindly using homomorphic addition
- **Precision:** Max error after encrypt -> add -> decrypt: **7.45e-09** (negligible)
- **Script:** `Encryption/ecyption_Tenseal_RNS.py`

### Stage 4.1: Dataset Distribution -- Complete

- **Input:** `Dataset/processed/mimic/mimic_ppwindowed_dataset.csv` (437 rows, 123 patients)
- **Method:** Stratified split by patient (not by row) to prevent data leakage
- **Output:** 3 hospitals x (X.npy + y.npy) in `Model_Training/Hospital_A/B/C/`
- **Split:** 41 patients per hospital (29 survived, 12 died each)
- **Data Shape:** X = (41, 8, 8) per hospital -- (patients, max_days, features)
- **Script:** `Data_Preprocessing/distribute_dataset.py`

### Stage 4.2: Local Hospital Training -- Complete

- **Method:** 5-Fold Stratified Cross-Validation (Phase 1) + Retrain on all data (Phase 2)
- **Config:** 300 epochs, LR=0.002, batch_size=8, BCEWithLogitsLoss (weighted), Adam, CosineAnnealingLR
- **Phase 1:** Provides honest generalization metrics via 5-Fold CV
- **Phase 2:** Retrains on all hospital data to produce `local_model.pth` for federated aggregation
- **Scripts:** `Model_Training/Hospital_A/train_hospital_A.py`, `Hospital_B/train_hospital_B.py`, `Hospital_C/train_hospital_C.py`

### Stage 4.3: Encrypt Local Weights -- Next

- Each hospital's `local_model.pth` will be encrypted using CKKS-RNS
- Encrypted weights sent to aggregation server (no raw weights leave the hospital)

### Stage 4.4: Federated Aggregation -- Pending

- Server receives encrypted weights from all 3 hospitals
- Performs homomorphic averaging (FedAvg) on ciphertexts
- Decrypts aggregated global model

### Stage 4.5: Evaluation & Comparison -- Pending

- Compare encrypted FL vs plaintext FL accuracy
- Measure cryptographic overhead

---

## Results

### LSTM Plaintext Baseline (5-Fold Cross-Validation)

#### Model Evolution

| Run | Architecture | Params | Accuracy | AUC-ROC | F1 (Died) | Issue |
|-----|-------------|--------|----------|---------|-----------|-------|
| 1 | LSTM(64, 2L) | 54,337 | 0.68 | 0.56 | 0.00 | Predicted all as "Survived" |
| 2 | LSTM(128, 3L) + weighted loss | 345,217 | 0.52 | 0.51 | 0.33 | Too many params, overfitting |
| 3 | LSTM(32, 1L) + weighted loss | 5,921 | 0.64 | 0.64 | 0.31 | Better, but single split unreliable |
| **4** | **LSTM(32, 1L) + weighted + CV + threshold** | **5,921** | **0.56** | **0.61** | **0.42** | **+ 5-Fold CV for robust metrics** |

#### Final Baseline (5-Fold CV)

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Average** |
|--------|--------|--------|--------|--------|--------|-------------|
| Accuracy | 0.680 | 0.640 | 0.640 | 0.542 | 0.625 | **0.625** |
| AUC-ROC | 0.778 | 0.714 | 0.728 | 0.555 | 0.588 | **0.673** |
| F1-Score | 0.429 | 0.526 | 0.609 | 0.421 | 0.471 | **0.491** |

### Local Hospital Training Results (5-Fold CV)

Each hospital trained independently on 41 patients (29 survived, 12 died):

| Hospital | CV Accuracy | CV AUC-ROC | CV F1-Score | Best Fold AUC |
|----------|------------|------------|-------------|---------------|
| A | 0.6333 | 0.5711 | 0.3133 | 0.8000 (Fold 5) |
| B | 0.5361 | 0.4333 | 0.2514 | 0.5833 (Fold 3) |
| C | 0.6583 | 0.4911 | 0.3733 | 0.6667 (Fold 2) |

> **Note:** These CV metrics reflect honest generalization on small per-hospital data (41 patients each). The federated aggregation step (4.4) will combine knowledge from all 3 hospitals to improve global performance.

---

## Requirements

```
Python 3.11+
pandas
numpy
torch (PyTorch with CUDA)
scikit-learn
tenseal >= 0.3.16
```

## How to Run

### Stage 1: Data Preprocessing
```bash
cd Data_Preprocessing
python Mimic_Raw_Combined.py      # Static feature extraction
python mimic_pp_day_wise.py       # 24h sliding window
```

### Stage 2: LSTM Baseline Training
```bash
python Model_Training/LSTM.py
```

### Stage 3: Encryption Verification
```bash
python Encryption/ecyption_Tenseal_RNS.py
```

### Stage 4.1: Distribute Dataset to Hospitals
```bash
python Data_Preprocessing/distribute_dataset.py
```

### Stage 4.2: Train Each Hospital Locally
```bash
python Model_Training/Hospital_A/train_hospital_A.py
python Model_Training/Hospital_B/train_hospital_B.py
python Model_Training/Hospital_C/train_hospital_C.py
```

### Stage 4.3-4.5: Encrypt, Aggregate, Evaluate (coming next)

---

## Dataset

This project uses the [MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/) -- a freely available subset of the MIMIC-III critical care database containing de-identified health data for ~100 patients.

## License

This project is for academic/research purposes. MIMIC-III data usage is governed by the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciii/1.4/).

---

## Changelog

| Date | Update |
|------|--------|
| Feb 15, 2026 | Stage 1 (Data Engineering) complete -- 437 samples from 123 patients |
| Feb 15, 2026 | Stage 2 (LSTM Baseline) complete -- 5-Fold CV: Acc=0.625, AUC=0.673, F1=0.491 |
| Feb 16, 2026 | Stage 3 (CKKS-RNS Encryption) complete -- TenSEAL verified, max error 7.45e-09 |
| Feb 16, 2026 | Project restructured into clean folder layout |
| Feb 16, 2026 | Stage 4.1 (Dataset Distribution) complete -- 3 hospitals x 41 patients (stratified) |
| Feb 16, 2026 | Stage 4.2 (Local Training) complete -- All 3 hospitals trained with 5-Fold CV |
