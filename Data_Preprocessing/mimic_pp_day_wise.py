import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Dataset", "raw", "mimic")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Dataset", "processed", "mimic")

# Vital Sign IDs (Same as before)
FEATURES = {
    'HeartRate': [211, 220045],
    'SysBP': [51, 442, 455, 6701, 220179, 220050],
    'RespRate': [618, 615, 220210, 224690],
    'Temp': [676, 223762, 678, 223761],
    'SpO2': [646, 220277],
    'Glucose': [807, 811, 1529, 225664, 220621, 226537]
}

# Flatten ID list for filtering
ALL_IDS = [id for sublist in FEATURES.values() for id in sublist]

def process_time_windows():
    print("Loading Core Tables...")
    # 1. Load Admissions to get ADMITTIME (Reference Point)
    df_adm = pd.read_csv(os.path.join(DATA_DIR, "ADMISSIONS.csv"),
                         usecols=['hadm_id', 'subject_id', 'admittime', 'hospital_expire_flag'])
    df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])
    
    # 2. Load Patients for Age/Gender
    df_pat = pd.read_csv(os.path.join(DATA_DIR, "PATIENTS.csv"), usecols=['subject_id', 'dob', 'gender'])
    df_pat['dob'] = pd.to_datetime(df_pat['dob'])
    
    # Merge basic info
    df_base = pd.merge(df_adm, df_pat, on='subject_id')
    
    print("Processing CHARTEVENTS in Windows (This creates more rows)...")
    
    # We will store multiple rows per patient here
    expanded_rows = []
    
    # Read CHARTEVENTS in chunks
    chunk_size = 1000000
    reader = pd.read_csv(os.path.join(DATA_DIR, "CHARTEVENTS.csv"),
                         usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'],
                         chunksize=chunk_size)

    # Dictionary to hold temp data: { HADM_ID: DataFrame_of_vitals }
    # Note: For efficiency in a demo, we'll process by grouping
    # But strictly, we need to map CHARTTIME relative to ADMITTIME
    
    # OPTIMIZED STRATEGY FOR MEMORY:
    # 1. Filter relevant rows
    # 2. Merge with ADMITTIME
    # 3. Calculate "Day Number"
    
    relevant_chunks = []
    for chunk in reader:
        # Keep only relevant vitals
        filtered = chunk[chunk['itemid'].isin(ALL_IDS)].dropna()
        if not filtered.empty:
            relevant_chunks.append(filtered)
            
    if not relevant_chunks:
        print("No matching chart events found!")
        return

    df_chart = pd.concat(relevant_chunks)
    df_chart['charttime'] = pd.to_datetime(df_chart['charttime'])
    
    # Merge Chart data with Admission Time to calculate "Time Since Admission"
    df_chart = pd.merge(df_chart, df_base[['hadm_id', 'admittime']], on='hadm_id', how='inner')
    
    # Calculate "Day Number" (0 = first 24h, 1 = 24-48h, etc.)
    df_chart['day_num'] = (df_chart['charttime'] - df_chart['admittime']).dt.total_seconds() // (24 * 3600)
    
    # Filter: Only keep first 7 days to avoid sparse long-tail data
    df_chart = df_chart[df_chart['day_num'].between(0, 7)]
    
    print(f"  Found {len(df_chart)} vital sign measurements across {df_chart['day_num'].nunique()} days.")

    # Pivot / Group By: (hadm_id + day_num) -> Mean Values
    # This is the magic step that multiplies your data
    grouped = df_chart.groupby(['hadm_id', 'day_num', 'itemid'])['valuenum'].mean().reset_index()
    
    # Now we reshape so Vitals are columns (HeartRate, SysBP...)
    # We iterate through unique hadm_id + day_num combinations
    
    # Map itemid to Feature Name
    id_to_name = {id: name for name, ids in FEATURES.items() for id in ids}
    grouped['feature_name'] = grouped['itemid'].map(id_to_name)
    
    # Pivot table: Index=(hadm_id, day_num), Columns=feature_name, Values=valuenum
    df_features = grouped.pivot_table(index=['hadm_id', 'day_num'], 
                                      columns='feature_name', 
                                      values='valuenum').reset_index()
    
    # Merge back with static data (Age, Gender, Target)
    final_df = pd.merge(df_features, df_base, on='hadm_id', how='inner')
    
    # Final cleanup (Age calculation using year diff to avoid overflow)
    final_df['age'] = final_df['admittime'].dt.year - final_df['dob'].dt.year
    final_df['gender'] = final_df['gender'].map({'M': 1, 'F': 0})
    
    # Fill missing values (forward fill then back fill, only on feature columns)
    feat_cols = list(FEATURES.keys())
    final_df[feat_cols] = final_df.groupby('hadm_id')[feat_cols].ffill().bfill().fillna(0)

    # Select Columns
    cols = ['hadm_id', 'day_num', 'hospital_expire_flag', 'age', 'gender'] + list(FEATURES.keys())
    # Keep only existing columns
    cols = [c for c in cols if c in final_df.columns]
    final_df = final_df[cols]
    
    print(f"--- SUCCESS! ---")
    print(f"Original Patients: {final_df['hadm_id'].nunique()}")
    print(f"New Expanded Dataset Size: {len(final_df)} rows")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "mimic_ppwindowed_dataset.csv")
    final_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    process_time_windows()