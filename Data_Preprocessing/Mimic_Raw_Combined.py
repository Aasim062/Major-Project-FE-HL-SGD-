import pandas as pd
import numpy as np
import os

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Dataset", "raw", "mimic")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Dataset", "processed", "mimic")

# --- CLINICAL FEATURES TO EXTRACT ---
# We use standard MIMIC-III ITEMIDs for both CareVue and MetaVision systems
FEATURES = {
    # VITALS (From CHARTEVENTS)
    'HeartRate': [211, 220045],
    'SysBP': [51, 442, 455, 6701, 220179, 220050],
    'RespRate': [618, 615, 220210, 224690],
    'Temp': [676, 223762, 678, 223761],
    'SpO2': [646, 220277],
    'Glucose': [807, 811, 1529, 225664, 220621, 226537],
    
    # LABS (From LABEVENTS)
    'Lactate': [50813, 52442],
    'Creatinine': [50912],
    'WBC': [51300, 51301],
    'Hemoglobin': [51222, 50811],
    'Platelets': [51265]
}

# Helper lists for filtering
CHART_IDS = [id for k, v in FEATURES.items() if k in ['HeartRate', 'SysBP', 'RespRate', 'Temp', 'SpO2', 'Glucose'] for id in v]
LAB_IDS = [id for k, v in FEATURES.items() if k in ['Lactate', 'Creatinine', 'WBC', 'Hemoglobin', 'Platelets'] for id in v]

def process_large_table(filename, target_ids, id_col, val_col):
    """
    Reads massive CSVs in chunks to save RAM.
    Returns the raw MEAN value for each feature per admission.
    """
    print(f"--- Processing {filename} in chunks... ---")
    aggregated_data = {} 
    
    # Read in chunks of 5 million rows
    chunk_size = 5000000 
    path = os.path.join(DATA_DIR, filename)
    
    for chunk in pd.read_csv(path, usecols=['hadm_id', id_col, val_col], chunksize=chunk_size):
        # Filter for relevant IDs
        relevant = chunk[chunk[id_col].isin(target_ids)].dropna()
        
        for _, row in relevant.iterrows():
            hadm_id = row['hadm_id']
            item_id = row[id_col]
            val = row[val_col]
            
            # Map ID to Feature Name
            feature_name = None
            for name, ids in FEATURES.items():
                if item_id in ids:
                    feature_name = name
                    break
            
            if feature_name:
                if hadm_id not in aggregated_data:
                    aggregated_data[hadm_id] = {k: [] for k in FEATURES.keys()}
                
                # Sanity Check: Ignore obvious sensor errors (e.g., Temp 0, HR 0)
                # But keep high/low values that might be real clinical events
                if val > 0: 
                    aggregated_data[hadm_id][feature_name].append(val)

    # Calculate Raw Mean
    final_rows = []
    for hadm_id, feats in aggregated_data.items():
        row = {'hadm_id': hadm_id}
        for name, values in feats.items():
            row[name] = np.mean(values) if values else np.nan
        final_rows.append(row)
        
    return pd.DataFrame(final_rows)

def main():
    print("STEP 1: Extracting Core Admissions & Patients...")
    
    # 1. Admissions (Target)
    df_adm = pd.read_csv(os.path.join(DATA_DIR, "ADMISSIONS.csv"), 
                         usecols=['subject_id', 'hadm_id', 'admittime', 'hospital_expire_flag', 'admission_type'])
    
    # 2. Patients (Demographics)
    df_pat = pd.read_csv(os.path.join(DATA_DIR, "PATIENTS.csv"), 
                         usecols=['subject_id', 'dob', 'gender'])
    
    # Merge Core
    df = pd.merge(df_adm, df_pat, on='subject_id', how='inner')

    # 3. Calculate Raw Age
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    # Raw Age calculation using year difference to avoid datetime overflow
    df['age'] = df['admittime'].dt.year - df['dob'].dt.year

    print("STEP 2: Extracting Severity (LOS)...")
    df_icu = pd.read_csv(os.path.join(DATA_DIR, "ICUSTAYS.csv"), usecols=['hadm_id', 'los'])
    # Sum LOS if patient had multiple ICU stays in one admission
    df_icu_sum = df_icu.groupby('hadm_id')['los'].sum().reset_index()
    df = pd.merge(df, df_icu_sum, on='hadm_id', how='left')

    print("STEP 3: Extracting Vitals (CHARTEVENTS)...")
    df_vitals = process_large_table("CHARTEVENTS.csv", CHART_IDS, 'itemid', 'valuenum')
    df = pd.merge(df, df_vitals, on='hadm_id', how='left')

    print("STEP 4: Extracting Labs (LABEVENTS)...")
    df_labs = process_large_table("LABEVENTS.csv", LAB_IDS, 'itemid', 'valuenum')
    df = pd.merge(df, df_labs, on='hadm_id', how='left')

    print("STEP 5: Extracting Comorbidities (DIAGNOSES_ICD)...")
    df_diag = pd.read_csv(os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv"), usecols=['hadm_id', 'icd9_code'])
    # Just counting raw number of codes per patient
    complexity = df_diag.groupby('hadm_id')['icd9_code'].nunique().reset_index()
    complexity.rename(columns={'icd9_code': 'num_comorbidities'}, inplace=True)
    df = pd.merge(df, complexity, on='hadm_id', how='left')

    print("STEP 6: Final Organization...")
    # Rename Target for clarity
    df.rename(columns={'hospital_expire_flag': 'target_mortality'}, inplace=True)
    
    # Reorder columns
    cols = ['subject_id', 'hadm_id', 'target_mortality', 'admittime', 'admission_type', 
            'age', 'gender', 'los', 'num_comorbidities'] + list(FEATURES.keys())
    
    # Keep only columns that actually exist (in case no labs found for some)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Save RAW file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "mimic_raw_combined.csv")
    df.to_csv(output_path, index=False)
    
    print(f"DONE! Raw combined file saved to: {output_path}")
    print(f"Total Records: {len(df)}")
    print("Columns included:", list(df.columns))

if __name__ == "__main__":
    main()