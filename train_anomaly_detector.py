import pyreadr
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# ============================================================
# LOAD FAULT-FREE TRAINING DATA
# ============================================================

normal_result = pyreadr.read_r(
    "data/Tennessee Eastman Process/TEP_FaultFree_Training.RData"
)

normal_df = normal_result["fault_free_training"]

print("\nFault-Free Training Shape:")
print(normal_df.shape)

# ============================================================
# LOAD FAULTY TEST DATA
# ============================================================

faulty_result = pyreadr.read_r(
    "data/Tennessee Eastman Process/TEP_Faulty_Testing.RData"
)

faulty_df = faulty_result["faulty_testing"]

print("\nFaulty Testing Shape:")
print(faulty_df.shape)

# ============================================================
# SAMPLE DATA FOR FASTER TRAINING
# ============================================================

normal_df = normal_df.sample(
    20000,
    random_state=42
)

faulty_df = faulty_df.sample(
    20000,
    random_state=42
)

# ============================================================
# FEATURES
# ============================================================

feature_cols = [
    col for col in normal_df.columns
    if col not in ["faultNumber", "simulationRun", "sample"]
]

X_normal = normal_df[feature_cols]
X_faulty = faulty_df[feature_cols]

# ============================================================
# SCALE DATA
# ============================================================

scaler = StandardScaler()

X_normal_scaled = scaler.fit_transform(X_normal)
X_faulty_scaled = scaler.transform(X_faulty)

# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Isolation Forest...")

model.fit(X_normal_scaled)

# ============================================================
# PREDICTIONS
# ============================================================

# IsolationForest outputs:
#  1  = normal
# -1 = anomaly

normal_pred = model.predict(X_normal_scaled)
faulty_pred = model.predict(X_faulty_scaled)

# ============================================================
# CREATE TRUE LABELS
# ============================================================

# Normal = 0
# Faulty = 1

y_true = (
    [0] * len(normal_pred)
    +
    [1] * len(faulty_pred)
)

# Convert predictions
y_pred = []

for pred in normal_pred:
    y_pred.append(0 if pred == 1 else 1)

for pred in faulty_pred:
    y_pred.append(0 if pred == 1 else 1)

# ============================================================
# RESULTS
# ============================================================

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Anomaly"]
))