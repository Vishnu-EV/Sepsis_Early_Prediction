"""
STEP 1: DATA LOADING & PREPROCESSING
=====================================
PhysioNet 2019 Sepsis Challenge Dataset
Each .psv file = one patient's hourly ICU readings
Columns: 40 features + SepsisLabel (0 or 1)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
DATA_DIR = "./data"          # folder containing .psv files
OUTPUT_DIR = "./processed"   # folder to save processed files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vital signs → used by TFT (time-series model)
VITAL_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

# Lab results + demographics → used by TabNet (tabular model)
LAB_COLS = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets'
]

DEMO_COLS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
# NOTE: ICULOS removed — it causes data leakage (longer ICU stay correlates
# with sepsis trivially, not clinically)

TARGET_COL = 'SepsisLabel'
TIME_STEPS = 12  # increased from 6 → 12 hours for better temporal coverage
                 # sepsis signals often appear 12-24 hours before onset


# ─────────────────────────────────────────
# 2. LOAD ALL PATIENT FILES
# ─────────────────────────────────────────
def load_all_patients(data_dir):
    """
    Reads all .psv files from the data folder.
    Returns a list of DataFrames, one per patient.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.psv')]
    print(f"Found {len(files)} patient files")

    all_dfs = []
    for i, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, sep='|')
        patient_id = fname.replace('.psv', '')
        df['PatientID'] = patient_id
        all_dfs.append(df)

        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i+1}/{len(files)} files...")

    print(f"Total patients loaded: {len(all_dfs)}")
    return all_dfs


# ─────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────
def handle_missing_values(df):
    """
    Strategy:
    - Forward fill: carry last known value forward (per patient over time)
    - Backward fill: fill remaining NaNs at the start
    - Column median fill: if still missing, use dataset-level median
    """
    feature_cols = VITAL_COLS + LAB_COLS + DEMO_COLS

    # Forward fill then backward fill within each patient's time series
    df[feature_cols] = df.groupby('PatientID')[feature_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # Fill any remaining NaNs with column median
    for col in feature_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):
    """
    Create additional clinical features:
    - Shock index: HR / SBP (ratio, high = poor prognosis)
    - Pulse pressure: SBP - DBP
    - Mean ICULOS per patient (how long they've been in ICU)
    """
    df['ShockIndex'] = df['HR'] / (df['SBP'] + 1e-6)
    df['PulsePressure'] = df['SBP'] - df['DBP']
    df['HR_MAP_ratio'] = df['HR'] / (df['MAP'] + 1e-6)
    return df


# ─────────────────────────────────────────
# 5. LABEL ASSIGNMENT (patient-level)
# ─────────────────────────────────────────
def get_patient_label(group):
    """
    If a patient ever has SepsisLabel=1 at any time step → label = 1
    Otherwise → label = 0
    """
    return int(group['SepsisLabel'].max())


# ─────────────────────────────────────────
# 6. BUILD TFT INPUT (Time-Series Sequences)
# ─────────────────────────────────────────
def build_tft_sequences(all_dfs, time_steps=TIME_STEPS):
    """
    For each patient, extract the last `time_steps` hours of vital signs.
    Shape: (num_patients, time_steps, num_vitals)
    """
    vital_features = VITAL_COLS + ['ShockIndex', 'PulsePressure', 'HR_MAP_ratio']
    X_seq = []
    y_seq = []

    for df in all_dfs:
        vitals = df[vital_features].values

        # Pad or trim to TIME_STEPS
        if len(vitals) >= time_steps:
            seq = vitals[-time_steps:]      # last N hours
        else:
            # Pad with zeros at the beginning if < time_steps rows
            pad = np.zeros((time_steps - len(vitals), len(vital_features)))
            seq = np.vstack([pad, vitals])

        label = int(df['SepsisLabel'].max())
        X_seq.append(seq)
        y_seq.append(label)

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    print(f"TFT sequences shape: {X_seq.shape}  Labels: {y_seq.shape}")
    return X_seq, y_seq


# ─────────────────────────────────────────
# 7. BUILD TABNET INPUT (Tabular per Patient)
# ─────────────────────────────────────────
def build_tabnet_features(all_dfs):
    """
    For each patient, aggregate features into a single row:
    - Mean, max, min, std of vitals and lab values
    - Last-row demographics
    """
    rows = []
    labels = []

    for df in all_dfs:
        row = {}

        # Aggregate vitals statistically (mean, max, min, std)
        for col in VITAL_COLS + ['ShockIndex', 'PulsePressure', 'HR_MAP_ratio']:
            if col in df.columns:
                row[f'{col}_mean'] = df[col].mean()
                row[f'{col}_max']  = df[col].max()
                row[f'{col}_min']  = df[col].min()
                row[f'{col}_std']  = df[col].std() if len(df) > 1 else 0.0

        # Aggregate labs (mean and last value)
        for col in LAB_COLS:
            if col in df.columns:
                row[f'{col}_mean'] = df[col].mean()
                row[f'{col}_last'] = df[col].iloc[-1]

        # Demographics from last row
        for col in DEMO_COLS:
            if col in df.columns:
                row[col] = df[col].iloc[-1]

        rows.append(row)
        labels.append(int(df['SepsisLabel'].max()))

    X_tab = pd.DataFrame(rows).fillna(0).values.astype(np.float32)
    y_tab = np.array(labels, dtype=np.float32)
    print(f"TabNet features shape: {X_tab.shape}  Labels: {y_tab.shape}")
    return X_tab, y_tab, pd.DataFrame(rows).columns.tolist()


# ─────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 55)

    # Load all patient files
    all_dfs = load_all_patients(DATA_DIR)

    # Merge into one big DataFrame for imputation
    print("\n[1] Combining all patient data...")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Combined shape: {combined.shape}")

    # Handle missing values
    print("\n[2] Handling missing values...")
    combined = handle_missing_values(combined)

    # Feature engineering
    print("\n[3] Engineering clinical features...")
    combined = engineer_features(combined)

    # Split back to per-patient list
    print("\n[4] Splitting back to per-patient...")
    grouped = [grp for _, grp in combined.groupby('PatientID')]

    # Build TFT sequences
    print("\n[5] Building TFT time-series sequences...")
    X_seq, y_seq = build_tft_sequences(grouped)

    # Build TabNet tabular features
    print("\n[6] Building TabNet tabular features...")
    X_tab, y_tab, tab_cols = build_tabnet_features(grouped)

    # Class balance check
    pos = y_seq.sum()
    neg = len(y_seq) - pos
    print(f"\n[7] Class balance — Sepsis: {int(pos)} | No Sepsis: {int(neg)}")
    print(f"  Positive rate: {pos/len(y_seq)*100:.1f}%")

    # Train/Val/Test split (70/15/15, stratified)
    # 3-way split ensures fusion layer trains on val, reports on unseen test
    print("\n[8] Train/Val/Test split (70/15/15, stratified)...")
    indices = np.arange(len(y_seq))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=y_seq)
    temp_labels = y_seq[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=temp_labels)

    X_seq_train, X_seq_val, X_seq_test = X_seq[train_idx], X_seq[val_idx], X_seq[test_idx]
    X_tab_train, X_tab_val, X_tab_test = X_tab[train_idx], X_tab[val_idx], X_tab[test_idx]
    y_train, y_val, y_test             = y_seq[train_idx], y_seq[val_idx], y_seq[test_idx]

    # Normalize
    print("\n[9] Normalizing features...")
    # Tabular scaler
    tab_scaler = StandardScaler()
    X_tab_train = tab_scaler.fit_transform(X_tab_train)
    X_tab_val   = tab_scaler.transform(X_tab_val)
    X_tab_test  = tab_scaler.transform(X_tab_test)

    # Sequence scaler
    n_train, t, f = X_seq_train.shape
    seq_scaler = StandardScaler()
    X_seq_train = seq_scaler.fit_transform(
        X_seq_train.reshape(-1, f)).reshape(n_train, t, f)
    X_seq_val   = seq_scaler.transform(
        X_seq_val.reshape(-1, f)).reshape(X_seq_val.shape[0], t, f)
    X_seq_test  = seq_scaler.transform(
        X_seq_test.reshape(-1, f)).reshape(X_seq_test.shape[0], t, f)

    # Save everything
    print("\n[10] Saving processed data...")
    np.save(f"{OUTPUT_DIR}/X_seq_train.npy", X_seq_train)
    np.save(f"{OUTPUT_DIR}/X_seq_val.npy",   X_seq_val)
    np.save(f"{OUTPUT_DIR}/X_seq_test.npy",  X_seq_test)
    np.save(f"{OUTPUT_DIR}/X_tab_train.npy", X_tab_train)
    np.save(f"{OUTPUT_DIR}/X_tab_val.npy",   X_tab_val)
    np.save(f"{OUTPUT_DIR}/X_tab_test.npy",  X_tab_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy",     y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy",       y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy",      y_test)

    with open(f"{OUTPUT_DIR}/tab_scaler.pkl", 'wb') as f:
        pickle.dump(tab_scaler, f)
    with open(f"{OUTPUT_DIR}/seq_scaler.pkl", 'wb') as f:
        pickle.dump(seq_scaler, f)
    with open(f"{OUTPUT_DIR}/tab_cols.pkl", 'wb') as f:
        pickle.dump(tab_cols, f)

    print("\n✅ Preprocessing complete! Files saved to ./processed/")
    print(f"   Train samples : {len(y_train)}")
    print(f"   Val   samples : {len(y_val)}")
    print(f"   Test  samples : {len(y_test)}")
    print(f"   Seq shape     : {X_seq_train.shape}")
    print(f"   Tab shape     : {X_tab_train.shape}")


if __name__ == "__main__":
    main()