# ============================================================
# LEAKAGE-SAFE LSTM SEQUENCE GENERATOR
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/tep_subset.csv")

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
# DYNAMIC RUN SPLIT
# ============================================================

all_runs = sorted(
    df["simulationRun"].unique()
)

split_index = int(
    len(all_runs) * 0.8
)

train_runs = all_runs[:split_index]
test_runs = all_runs[split_index:]

print("\nTrain Runs:", len(train_runs))
print("Test Runs:", len(test_runs))

# ============================================================
# CREATE TRAIN SEQUENCES
# ============================================================

for fault in df["faultNumber"].unique():

    fault_df = df[
        df["faultNumber"] == fault
    ]

    # -------------------------
    # TRAIN RUNS
    # -------------------------

    train_df = fault_df[
        fault_df["simulationRun"].isin(train_runs)
    ]

    for run_id in train_df["simulationRun"].unique():

        run_data = train_df[
            train_df["simulationRun"] == run_id
        ]

        X_run = run_data[feature_cols].values
        y_run = run_data["faultNumber"].values

        for i in range(
            len(run_data) - sequence_length
        ):

            X_seq = X_run[
                i:i + sequence_length
            ]

            y_seq = y_run[
                i + sequence_length
            ]

            X_train.append(X_seq)
            y_train.append(y_seq)

    # -------------------------
    # TEST RUNS
    # -------------------------

    test_df = fault_df[
        fault_df["simulationRun"].isin(test_runs)
    ]

    for run_id in test_df["simulationRun"].unique():

        run_data = test_df[
            test_df["simulationRun"] == run_id
        ]

        X_run = run_data[feature_cols].values
        y_run = run_data["faultNumber"].values

        for i in range(
            len(run_data) - sequence_length
        ):

            X_seq = X_run[
                i:i + sequence_length
            ]

            y_seq = y_run[
                i + sequence_length
            ]

            X_test.append(X_seq)
            y_test.append(y_seq)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

print("\nTrain Shapes:")
print(X_train.shape)
print(y_train.shape)

print("\nTest Shapes:")
print(X_test.shape)
print(y_test.shape)

# ============================================================
# SAVE
# ============================================================

np.save("X_train_lstm.npy", X_train)
np.save("y_train_lstm.npy", y_train)

np.save("X_test_lstm.npy", X_test)
np.save("y_test_lstm.npy", y_test)

print("\nLeakage-safe LSTM datasets saved!")