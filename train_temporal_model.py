import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import os
if not os.path.exists("models/"):
    import download_models
if not os.path.exists("data/"):
    import download_dataset



# ============================================================
# LOAD TEMPORAL DATASET
# ============================================================
#https://drive.google.com/file/d/1-iNc8QXjQo29-Az-5pkPmhskMt6J7AzL/view?usp=drive_link
df = pd.read_csv("data/tep_temporal_features.csv")

print("Dataset Shape:", df.shape)

# ============================================================
# FEATURES
# ============================================================

X = df.drop(columns=[
    "faultNumber",
    "simulationRun",
    "sample"
])

y = df["faultNumber"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Temporal Random Forest...\n")

model.fit(X_train, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred
))

print("\nConfusion Matrix:\n")

print(confusion_matrix(
    y_test,
    y_pred
))