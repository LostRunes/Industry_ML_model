# Industrial AI Monitoring System (Tennessee Eastman Process)

An enterprise-grade, real-time autonomous industrial monitoring dashboard designed to classify faults, detect anomalies, diagnose process drift, and predict pre-failure signatures. Built using the classical **Tennessee Eastman Process (TEP)** benchmark dataset and a hybrid ML/DL architecture.

---

## 🌟 Key Features

### 1. Hybrid Machine Learning & Deep Learning Engine
* **Multiclass Fault Classifier**: A highly regularized Random Forest model trained on TEP features to categorize active faults instantly.
* **Process Anomaly Detector**: An Isolation Forest model trained on normal operating baseline data to isolate anomalies from standard operations.
* **LSTM Deep Sequence Predictor**: A temporal neural network model using a sliding 10-frame sequence buffer to classify process trajectories and predict evolving fault patterns with deep confidence tracking.

### 2. Root Cause Analysis (RCA) Panel
* **Cached Operating Baseline**: Loads and caches baseline measurements under standard fault-free operations.
* **Sensor Deviation Ranking**: Dynamically calculates absolute deviations of all 52 process channels against the baseline, ranking the top 10 most abnormal process indicators.
* **Telemetry Bar Chart**: Instantly plots deviations to give control room operators immediate troubleshooting indicators.

### 3. Sensor Abnormality Heatmap
* **Spatial Telemetry Projection**: Reshapes the 52-dimensional process deviation vector into a clean `4 x 13` grid representing the physical distribution layout of TEP sensors.
* **Turbo-Scale Colormap**: Employs high-fidelity, transparent Plotly heatmaps. Brighter regions instantly indicate active thermal, pressure, or flow abnormalities during a fault event.

### 4. Predictive Maintenance & Early Warning System
* **Rising Confidence Monitor**: Automatically tracks the rolling trend of LSTM fault probabilities.
* **Trend Analysis Alert**: Triggers a **Predictive Maintenance Alert** before full classification is logged, warning operators when pre-failure instabilities emerge (rising trend above 60% confidence).

### 5. High-Fidelity Fault Replay Simulator
* **Interactive Control Selector**: A sidebar simulator that allows operators to select and inject frame-by-frame sequential replay measurements for specific faults.
* **HMI-like rendering**: Runs at a smooth **5 FPS (200ms auto-refresh loop)** server-side to simulate dynamic control room panel behaviors.

---

## 🔬 The LSTM Evolutionary Journey (Leakage vs. Scaled Industrial)

The core deep learning sequence predictor evolved through three distinct iterations, representing a professional journey from standard prototyping to scientifically rigorous, production-scale ML engineering.

### Performance & Structural Comparison

| Model / Pipeline Version | Preprocessing & Splitting Strategy | Leakage Status | Test Accuracy | Macro F1-Score | Data Scope / Total Sequences |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Historical / Leakage-Prone** <br>`train_lstm_model_with_leakage.py` | Overlapping sliding-window sequences generated *before* random train-test splitting; feature scaling fit on the entire dataset. | 🛑 **High Leakage** <br>(Future information leaked via overlap & scaling) | **97.66%** <br>*(Artificially inflated)* | **97.60%** | `11,660` sequences (Subset) |
| **2. Leakage-Safe (Subset)** <br>`train_lstm_model.py` | Simulation runs sorted and split *before* creating sequences (80% train runs, 20% test runs). Feature scaling fit exclusively on training data. | ✅ **Leakage-Safe** <br>(Strict separation of runs) | **84.55%** <br>*(Scientifically valid)* | **83.00%** | `11,660` sequences (Subset) |
| **3. Industrial-Scale (Full)** <br>`train_lstm_model_full.py` | Dynamic chronological splitting on the full RData dataset. Memory-optimized downsampling (first 50 runs per fault), `float32` precision casting, and training-set shuffling. | ✅ **Leakage-Safe** <br>(Production scale) | **88.00%** <br>*(Highly robust)* | **89.00%** | **`490,000` sequences** <br>(Full Dataset) |

### Why This Evolution Matters (Interview Value)
* **The Danger of Data Leakage in Time Series:** Generating sliding-window sequences across simulation boundaries before splitting means adjacent sequences share 9 out of 10 data frames. In a random split, a test sequence's near-identical neighbors will almost certainly end up in the training set. This creates massive data leakage, resulting in an artificially perfect but fragile model.
* **The Chronological Split Fix:** By partitioning the simulation runs first (e.g., training only on runs 1-40, and testing only on runs 41-50), the model is evaluated on completely unseen plant runs, providing a true reflection of control-room performance.
* **Paging & Memory Optimization:** Loading the 5-million-row RData file in Python easily triggers process-level `MemoryError` exceptions. We engineered a **memory-safe downsampling pipeline** that filters to a balanced subset of runs and casts float features to `float32` immediately, cutting RAM usage in half while maintaining state-of-the-art predictive performance on **490,000 sequences**.

---

## 📂 Project Structure

```bash
industry-model/
│
├── app/
│   └── dashboard.py                  # Streamlit UI dashboard with premium dark-cyber styling
│
├── data/                             # Subfolder holding project data
│   ├── tep_subset.csv                # TEP dataset subset
│   ├── TEP_Faulty_Training.RData     # FULL 5-million-row industrial RData dataset
│   ├── tep_replay.csv                # Custom 10,000-frame sequential fault replay dataset
│   ├── incident_logs.csv             # Autonomous CSV logging engine for plant anomalies
│   ├── presets.json                  # Pre-computed simulation preset states
│   └── ...
│
├── models/                           # Pre-trained ML/DL models & scalers
│   ├── tep_fault_classifier.pkl      # Random Forest Classifier
│   ├── tep_scaler.pkl
│   ├── lstm_fault_model.keras        # Leakage-Safe LSTM Model (Subset)
│   ├── lstm_label_encoder.pkl
│   ├── lstm_fault_model_full.keras   # Scaled Industrial LSTM Model (Full)
│   ├── lstm_label_encoder_full.pkl
│   ├── lstm_fault_model_with_leakage.keras # Historical Leakage Model (Subset)
│   └── ...
│
├── download_dataset.py               # Utility downloading the base TEP subset
├── download_models.py                # Utility downloading pre-trained network model binaries
├── prepare_replay_data.py            # Prepares temporal sequence replay data
│
├── create_lstm_sequences.py          # Leakage-Safe Sequence Generator (Subset)
├── train_lstm_model.py               # Leakage-Safe LSTM Trainer (Subset)
│
├── create_lstm_sequences_full.py     # Memory-Safe Sequence Generator (Full RData)
├── train_lstm_model_full.py          # Scaled LSTM Trainer (Full RData, Batch Size 512)
│
├── create_lstm_sequences_with_leakage.py # Historical Generator (with Data Leakage)
├── train_lstm_model_with_leakage.py  # Historical Trainer (with Data Leakage)
│
├── train_tep_model.py                # Random Forest model training script
├── train_anomaly_detector.py         # Isolation Forest model training script
├── requirements.txt                  # Python dependency specifications
└── README.md                         # Project README documentation
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment:
```bash
# Navigate to the project directory
cd industry-model

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate    # On Linux/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Historical model (with Data Leakage)
To reproduce the historic 97.66% data-leakage benchmark:
```bash
python create_lstm_sequences_with_leakage.py
python train_lstm_model_with_leakage.py
```

### 4. Run the Leakage-Safe model (Subset)
To run the leakage-free, scientifically rigorous subset model:
```bash
python create_lstm_sequences.py
python train_lstm_model.py
```

### 5. Run the Scaled Industrial model (Full Dataset)
To preprocess and train on the massive 5-million-row industrial RData dataset:
```bash
python create_lstm_sequences_full.py
python train_lstm_model_full.py
```

### 6. Run the Premium Dashboard
Launch the Streamlit web dashboard:
```bash
streamlit run app/dashboard.py
```
Open the local browser link rendered in your terminal (typically `http://localhost:8501`) to access the industrial control console.

---

## 🎨 Premium UX Style Guide
The dashboard uses a custom-developed **Dark Cybernetic Industrial Theme**:
* **Glassmorphic Panels**: Custom card wrappers with dark transparent backdrops, delicate glowing borders, and neon telemetry status accents.
* **Modern Typography**: Telemetric Outfit font blended with JetBrains Mono code panels.
* **Neon Alert Indication**: Glowing crimson red warnings for anomalies and glowing cyan/teal alerts for stable states.
* **Anti-Fighting Logic**: Integrated state-conflict deactivators on preset controls to automatically freeze active sequential replays when static presets are selected, ensuring high operational reliability.
