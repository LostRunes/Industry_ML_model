# ============================================================
# LEAKAGE-SAFE LSTM SEQUENCE GENERATOR (FULL INDUSTRIAL VERSION)
# ============================================================

import pyreadr
import pandas as pd
import numpy as np
import gc

# ============================================================
# LOAD FULL DATA
# ============================================================

print("Loading full industrial TEP dataset from RData...")
result = pyreadr.read_r("data/Tennessee Eastman Process/TEP_Faulty_Training.RData")
df = result["faulty_training"]

print("Original dataset shape:", df.shape)

# Free up dictionary memory immediately
del result
gc.collect()

# ============================================================
# DYNAMIC RUN SELECT & DOWNSAMPLE
# ============================================================

print("\nDownsampling simulation runs to ensure memory safety...")
keep_runs = []
for fault in sorted(df["faultNumber"].unique()):
    # Get all simulation runs for this fault
    fault_df = df[df["faultNumber"] == fault]
    fault_runs = sorted(fault_df["simulationRun"].unique())
    # Select first 50 runs for each fault to ensure equal, balanced representation
    keep_runs.extend(fault_runs[:50])

df = df[df["simulationRun"].isin(keep_runs)].copy()
print("Downsampled dataset shape:", df.shape)
gc.collect()

# ============================================================
# CONVERT TO FLOAT32 FOR MEMORY EFFICIENCY
# ============================================================

print("\nConverting float columns to float32...")
float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].astype("float32")
gc.collect()

# ============================================================
# SORT TEMPORALLY
# ============================================================

df = df.sort_values(
    by=["faultNumber", "simulationRun", "sample"]
)

# ============================================================
# FEATURES
# ============================================================

feature_cols = [
    col for col in df.columns
    if col not in [
        "faultNumber",
        "simulationRun",
        "sample"
    ]
]

# ============================================================
# PARAMETERS
# ============================================================

sequence_length = 10

X_train = []
y_train = []

X_test = []
y_test = []

# ============================================================
# CREATE SEQUENCES (LEAKAGE-SAFE SPLIT)
# ============================================================

print("\nGenerating sequences separately for train and test runs...")

for fault in sorted(df["faultNumber"].unique()):
    print(f"Processing fault: {fault}...")
    fault_df = df[df["faultNumber"] == fault]
    
    # Get all sorted unique runs for this fault
    unique_runs = sorted(fault_df["simulationRun"].unique())
    
    # 80/20 train/test split of runs
    split_idx = int(len(unique_runs) * 0.8)
    train_runs = unique_runs[:split_idx]
    test_runs = unique_runs[split_idx:]
    
    # -------------------------
    # TRAIN RUNS
    # -------------------------
    train_df = fault_df[fault_df["simulationRun"].isin(train_runs)]
    for run_id in train_df["simulationRun"].unique():
        run_data = train_df[train_df["simulationRun"] == run_id]
        X_run = run_data[feature_cols].values
        y_run = run_data["faultNumber"].values
        
        for i in range(len(run_data) - sequence_length):
            X_train.append(X_run[i:i + sequence_length])
            y_train.append(y_run[i + sequence_length])
            
    # -------------------------
    # TEST RUNS
    # -------------------------
    test_df = fault_df[fault_df["simulationRun"].isin(test_runs)]
    for run_id in test_df["simulationRun"].unique():
        run_data = test_df[test_df["simulationRun"] == run_id]
        X_run = run_data[feature_cols].values
        y_run = run_data["faultNumber"].values
        
        for i in range(len(run_data) - sequence_length):
            X_test.append(X_run[i:i + sequence_length])
            y_test.append(y_run[i + sequence_length])

# ============================================================
# CONVERT TO NUMPY ARRAYS WITH FLOAT32
# ============================================================

print("\nConverting sequences to NumPy arrays...")
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

print("\nTrain Shapes:")
print(X_train.shape)
print(y_train.shape)

print("\nTest Shapes:")
print(X_test.shape)
print(y_test.shape)

# ============================================================
# SAVE AS FULL DATASETS
# ============================================================

print("\nSaving sequences...")
np.save("X_train_lstm_full.npy", X_train)
np.save("y_train_lstm_full.npy", y_train)

np.save("X_test_lstm_full.npy", X_test)
np.save("y_test_lstm_full.npy", y_test)

print("\nLeakage-safe Full LSTM datasets saved successfully!")
