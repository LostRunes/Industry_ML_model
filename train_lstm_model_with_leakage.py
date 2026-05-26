# ============================================================
# TEP LSTM MODEL TRAINING (WITH DATA LEAKAGE - HISTORIC VERSION)
# ============================================================
# WARNING: This script contains data leakage because scaling is performed
# before splitting, and the train-test split is randomly performed over 
# overlapping sequences from the same simulation runs.
# This results in an artificially inflated test accuracy (~97.66%).

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

import joblib

# ============================================================
# LOAD DATA
# ============================================================

X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")

print("Original X Shape:", X.shape)

# ============================================================
# SCALE FEATURES
# ============================================================

# Reshape for scaling
samples, timesteps, features = X.shape

X_reshaped = X.reshape(-1, features)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_reshaped)

# Reshape back
X = X_scaled.reshape(
    samples,
    timesteps,
    features
)

print("Scaled X Shape:", X.shape)

# ============================================================
# ENCODE LABELS
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ============================================================
# BUILD LSTM MODEL
# ============================================================

model = Sequential([

    Input(shape=(
        X.shape[1],
        X.shape[2]
    )),

    LSTM(
        128,
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(64),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        y_categorical.shape[1],
        activation="softmax"
    )
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# TRAIN
# ============================================================

print("\nTraining LSTM with leakage...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=128
)

# ============================================================
# EVALUATE
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy (Leakage-Inflated):")
print(accuracy)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred_probs = model.predict(X_test)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

print("\nClassification Report (Leakage-Inflated):\n")

print(classification_report(
    y_true,
    y_pred
))

# ============================================================
# SAVE MODEL
# ============================================================

import os
os.makedirs("models", exist_ok=True)

model.save("models/lstm_fault_model_with_leakage.keras")
print("\nLSTM model with leakage saved!")

joblib.dump(
    encoder,
    "models/lstm_label_encoder_with_leakage.pkl"
)

print("\nLSTM label encoder with leakage saved!")
