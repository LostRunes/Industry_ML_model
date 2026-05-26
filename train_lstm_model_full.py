import numpy as np

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
# LOAD FULL DATA
# ============================================================

print("Loading full industrial LSTM datasets...")
X_train = np.load("X_train_lstm_full.npy")
y_train = np.load("y_train_lstm_full.npy")

X_test = np.load("X_test_lstm_full.npy")
y_test = np.load("y_test_lstm_full.npy")

print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")

# ============================================================
# SCALE FEATURES (NO LEAKAGE)
# ============================================================

print("\nScaling features (fitting only on train dataset)...")
train_samples, timesteps, features = X_train.shape
test_samples = X_test.shape[0]

# Reshape
X_train_reshaped = X_train.reshape(-1, features)
X_test_reshaped = X_test.reshape(-1, features)

# Fit only on train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape back
X_train = X_train_scaled.reshape(train_samples, timesteps, features)
X_test = X_test_scaled.reshape(test_samples, timesteps, features)

print("Scaled Train Shape:", X_train.shape)
print("Scaled Test Shape:", X_test.shape)

# ============================================================
# ENCODE LABELS
# ============================================================

print("\nEncoding labels...")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

y_train = to_categorical(y_train_encoded)
y_test = to_categorical(y_test_encoded)

# ============================================================
# SHUFFLE TRAINING DATA (REPRESENTATIVE VALIDATION)
# ============================================================

print("\nShuffling training dataset for representative validation split...")
indices = np.arange(len(X_train))
np.random.seed(42)
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\nBuilding industrial LSTM model...")
model = Sequential([
    Input(shape=(
        X_train.shape[1],
        X_train.shape[2]
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
        y_train.shape[1],
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

print("\nTraining industrial LSTM model...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    shuffle=True,
    epochs=10,
    batch_size=512
)

# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating model on test dataset...")
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nFull Test Accuracy:")
print(accuracy)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nIndustrial Classification Report:\n")
print(classification_report(
    y_true,
    y_pred
))

# ============================================================
# SAVE MODEL & ASSETS
# ============================================================

import os
os.makedirs("models", exist_ok=True)

model.save("models/lstm_fault_model_full.keras")
print("\nIndustrial LSTM model saved to models/lstm_fault_model_full.keras!")

joblib.dump(
    encoder,
    "models/lstm_label_encoder_full.pkl"
)
print("Industrial LSTM label encoder saved to models/lstm_label_encoder_full.pkl!")
