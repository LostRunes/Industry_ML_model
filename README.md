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

## 📂 Project Structure

```bash
industry-model/
│
├── app/
│   └── dashboard.py          # Streamlit UI dashboard with premium dark-cyber styling
│
├── data/                     # Subfolder holding project data (organized & cleaned up)
│   ├── tep_subset.csv        # TEP dataset subset
│   ├── tep_replay.csv        # Custom 10,000-frame sequential fault replay dataset
│   ├── incident_logs.csv     # Autonomous CSV logging engine for plant anomalies
│   ├── presets.json          # Pre-computed simulation preset states (Normal, Fault 3, etc.)
│   ├── feature_importances.csv
│   └── pca_background.csv
│
├── models/                   # Pre-trained ML/DL models & scalers
│   ├── tep_fault_classifier.pkl
│   ├── tep_scaler.pkl
│   ├── anomaly_detector.pkl
│   ├── anomaly_scaler.pkl
│   ├── lstm_fault_model.keras
│   └── lstm_label_encoder.pkl
│
├── download_dataset.py       # Utility downloading the base TEP subset
├── download_models.py        # Utility downloading pre-trained network model binaries
├── prepare_replay_data.py    # Prepares temporal sequence replay data
├── train_tep_model.py        # Random Forest model training script
├── train_anomaly_detector.py # Isolation Forest model training script
├── train_lstm_model.py       # LSTM sequence classifier training script
├── create_temporal_features.py # Rolling temporal sliding-window feature extractor
├── train_temporal_model.py   # Training script for temporal ML features
│
├── requirements.txt          # Python dependency specifications
└── README.md                 # Project README documentation
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment:
```bash
# Clone the repository and navigate to the project directory
cd industry-model

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate    # On Linux/macOS
```

### 2. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Download & Prepare Datasets/Models
Execute the download and preparation scripts:
```bash
# Download datasets
python download_dataset.py

# Download ML/DL model binaries
python download_models.py

# Prepare sequential replay dataset
python prepare_replay_data.py
```

### 4. Run the Premium Dashboard
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

---

## 📊 ML Architecture & Development
For developers looking to retrain the underlying intelligence models:
* Run `train_tep_model.py` to rebuild the 150-estimator Random Forest classifier.
* Run `train_lstm_model.py` to adjust sequential sliding-window neural weights.
* Run `create_temporal_features.py` to extract customized sliding window features (std, variance, rate-of-change) and train them on `train_temporal_model.py`.
