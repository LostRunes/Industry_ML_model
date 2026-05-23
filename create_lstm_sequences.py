#sequence dataset generator for lstm
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
    by=["simulationRun", "sample"]
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

X_sequences = []
y_sequences = []

# ============================================================
# CREATE SEQUENCES
# ============================================================

for run_id in df["simulationRun"].unique():

    run_data = df[
        df["simulationRun"] == run_id
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

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print("X shape:", X_sequences.shape)
print("y shape:", y_sequences.shape)

# ============================================================
# SAVE
# ============================================================

np.save("X_lstm.npy", X_sequences)
np.save("y_lstm.npy", y_sequences)

print("\nLSTM sequences saved!")