#streamlit run app/dashboard.py
import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from datetime import datetime

from streamlit_option_menu import option_menu

from tensorflow.keras.models import load_model
from collections import deque



if not os.path.exists("models/"):
    import download_models
if not os.path.exists("data/"):
    import download_dataset


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial AI Monitoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: white;
}

.metric-box {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS & PRE-COMPUTED DATA
# ============================================================

classifier = joblib.load("models/tep_fault_classifier.pkl")
classifier_scaler = joblib.load("models/tep_scaler.pkl")

anomaly_model = joblib.load("models/anomaly_detector.pkl")
anomaly_scaler = joblib.load("models/anomaly_scaler.pkl")

pca_model = joblib.load("models/pca_model.pkl")

lstm_model = load_model(
    "models/lstm_fault_model.keras"
)

lstm_encoder = joblib.load(
    "models/lstm_label_encoder.pkl"
)
# ============================================================
# FEATURE LIST
# ============================================================

feature_names = [
    *(f"xmeas_{i}" for i in range(1, 42)),
    *(f"xmv_{i}" for i in range(1, 12))
]

# ============================================================
# NORMAL BASELINE
# ============================================================

@st.cache_data
def get_normal_baseline():
    try:
        df = pd.read_csv("data/tep_subset.csv")
        return df[feature_names].mean()
    except Exception as e:
        try:
            with open("data/presets.json") as f:
                presets = json.load(f)
            return pd.Series(presets["0"], index=feature_names)
        except Exception:
            return pd.Series(0.0, index=feature_names)

normal_baseline = get_normal_baseline()

# ============================================================
# LOAD REPLAY DATASET
# ============================================================

@st.cache_data
def load_replay_dataset():
    try:
        return pd.read_csv("data/tep_replay.csv")
    except Exception:
        return pd.DataFrame()

replay_df = load_replay_dataset()

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "input_data" not in st.session_state:
    try:
        with open("data/presets.json") as f:
            presets_data = json.load(f)
        st.session_state.input_data = np.array(presets_data["0"])
    except Exception:
        st.session_state.input_data = np.random.normal(50, 10, len(feature_names))

if "history" not in st.session_state:
    st.session_state.history = []

if "anomaly_active" not in st.session_state:
    st.session_state.anomaly_active = False

if "last_logged_fault" not in st.session_state:
    st.session_state.last_logged_fault = None

if "sequence_buffer" not in st.session_state:

    st.session_state.sequence_buffer = deque(
        maxlen=10
    )

if "probability_history" not in st.session_state:
    st.session_state.probability_history = []

if "fault_history" not in st.session_state:
    st.session_state.fault_history = []

if "replay_index" not in st.session_state:
    st.session_state.replay_index = 0

if "replay_active" not in st.session_state:
    st.session_state.replay_active = False

# ============================================================
# SIDEBAR & PRESETS
# ============================================================

with st.sidebar:

    selected = option_menu(
        menu_title="Industrial AI System",
        options=[
            "Process Monitoring",
            "Fault Detection",
            "Anomaly Analysis"
        ],
        icons=[
            "activity",
            "cpu",
            "graph-up"
        ],
        default_index=0
    )
    
    st.markdown("---")
    st.subheader("Simulation Presets")
    
    try:
        with open("data/presets.json") as f:
            presets = json.load(f)
            
        if st.button("Normal State"):
            st.session_state.input_data = np.array(presets["0"])
            st.session_state.anomaly_active = False
            st.session_state.last_logged_fault = None
            st.rerun()
            
        if st.button("Fault 3 Simulation"):
            st.session_state.input_data = np.array(presets["3"])
            st.session_state.anomaly_active = False
            st.session_state.last_logged_fault = None
            st.rerun()
            
        if st.button("Fault 9 Simulation"):
            st.session_state.input_data = np.array(presets["9"])
            st.session_state.anomaly_active = False
            st.session_state.last_logged_fault = None
            st.rerun()
            
        if st.button("High-Risk Process State"):
            st.session_state.input_data = np.array(presets["1"])
            st.session_state.anomaly_active = False
            st.session_state.last_logged_fault = None
            st.rerun()
    except Exception as e:
        st.error(f"Error loading presets: {e}")
        
    if st.button("Generate Random State"):
        st.session_state.input_data = np.random.normal(50, 10, len(feature_names))
        st.session_state.anomaly_active = False
        st.session_state.last_logged_fault = None
        st.rerun()

    st.markdown("---")
    st.subheader("Fault Replay Simulator")

    selected_fault = st.selectbox(
        "Select Fault",
        sorted(replay_df["faultNumber"].unique()) if not replay_df.empty else [0]
    )

    start_replay = st.button(
        "Start Replay"
    )

# ============================================================
# START REPLAY LOGIC & FILTER FAULT DATA
# ============================================================

if start_replay:
    st.session_state.replay_active = True
    st.session_state.replay_index = 0
    st.session_state.sequence_buffer.clear()

if not replay_df.empty:
    fault_data = replay_df[
        replay_df["faultNumber"] == selected_fault
    ].reset_index(drop=True)
else:
    fault_data = pd.DataFrame()

# ============================================================
# RUN AI ANALYSIS (CONTINUOUS)
# ============================================================

input_array = np.array(
    st.session_state.input_data
).reshape(1, -1)

# 1. Classification
scaled_classifier_input = classifier_scaler.transform(input_array)
probabilities = classifier.predict_proba(scaled_classifier_input)[0]
predicted_fault = classifier.predict(scaled_classifier_input)[0]
confidence = np.max(probabilities)

# 2. Anomaly Detection
scaled_anomaly_input = anomaly_scaler.transform(input_array)
anomaly_prediction = anomaly_model.predict(scaled_anomaly_input)[0]

# 3. Process Health Score
health_score = max(
    0.0,
    100.0 - (confidence * 100.0 if anomaly_prediction == -1 else 0.0)
)

# 4. Process Drift Diagnosis
drift_score = np.mean(np.abs(scaled_anomaly_input))
if drift_score < 1.2:
    drift_status = "LOW DRIFT"
    drift_color = "#00FFFF" # Cyan
elif drift_score < 2.2:
    drift_status = "MODERATE DRIFT"
    drift_color = "#FFA500" # Orange
else:
    drift_status = "HIGH DRIFT"
    drift_color = "#FF0000" # Red

# 5. Live PCA Projection
live_scaled = classifier_scaler.transform(input_array)
live_pca = pca_model.transform(live_scaled)[0]

# ====================================================
# LSTM SEQUENCE PREDICTION
# ====================================================

lstm_prediction = "Waiting for sequence..."
lstm_confidence = 0.0
early_warning = False
if len(st.session_state.sequence_buffer) == 10:

    sequence_array = np.array(
        st.session_state.sequence_buffer
    )

    sequence_array = sequence_array.reshape(
        1,
        10,
        len(feature_names)
    )

    # SCALE SEQUENCE
    reshaped = sequence_array.reshape(
        -1,
        len(feature_names)
    )

    reshaped = classifier_scaler.transform(
        reshaped
    )

    sequence_array = reshaped.reshape(
        1,
        10,
        len(feature_names)
    )

    lstm_probs = lstm_model.predict(
        sequence_array,
        verbose=0
    )

    # Store probability history
    if "probability_history" not in st.session_state:

        st.session_state.probability_history = []

    current_probs = lstm_probs[0]

    st.session_state.probability_history.append(
        current_probs
    )

    # Keep only latest 30 frames
    st.session_state.probability_history = (
        st.session_state.probability_history[-30:]
    )



    lstm_class = np.argmax(lstm_probs)
    lstm_confidence = float(np.max(lstm_probs))

    lstm_prediction = (
        lstm_encoder.inverse_transform(
            [lstm_class]
        )[0]
    )

    st.session_state.fault_history.append(
        float(lstm_confidence)
    )

    # Keep latest 30 frames
    st.session_state.fault_history = (
        st.session_state.fault_history[-30:]
    )

    if len(st.session_state.fault_history) >= 5:

        recent = st.session_state.fault_history[-5:]

        # Check for rising trend
        if (
            recent[-1] > recent[0]
            and recent[-1] > 0.60
        ):

            early_warning = True

# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

current_values = pd.Series(
    st.session_state.input_data,
    index=feature_names
)

deviations = abs(
    current_values - normal_baseline
)

top_deviations = deviations.sort_values(
    ascending=False
).head(10)

# ============================================================
# FULL SENSOR DEVIATION VECTOR
# ============================================================

sensor_deviation_df = pd.DataFrame({

    "Sensor": feature_names,

    "Deviation": deviations.values

})

# ============================================================
# INCIDENT LOGGING ENGINE
# ============================================================

if anomaly_prediction == -1:
    if not st.session_state.anomaly_active or st.session_state.last_logged_fault != predicted_fault:
        log_file = "data/incident_logs.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = pd.DataFrame([{
            "Timestamp": timestamp,
            "Predicted Fault": f"Fault {predicted_fault}",
            "Confidence": f"{confidence * 100:.2f}%",
            "Health Score": f"{health_score:.1f}%",
            "Drift Status": drift_status
        }])
        
        if not os.path.exists(log_file):
            log_data.to_csv(log_file, index=False)
        else:
            log_data.to_csv(log_file, mode='a', header=False, index=False)
            
        st.session_state.anomaly_active = True
        st.session_state.last_logged_fault = predicted_fault
else:
    st.session_state.anomaly_active = False
    st.session_state.last_logged_fault = None

# ============================================================
# HEADER & ALERTS
# ============================================================

st.title("Industrial Process Monitoring System")

st.markdown("""
Real-time autonomous industrial fault classification and anomaly detection
using Tennessee Eastman Process benchmark data.
""")

# ============================================================
# REPLAY STATUS
# ============================================================

st.subheader("Replay Status")

if st.session_state.replay_active:

    st.success(
        f"Replaying Fault {selected_fault}"
    )

    st.write(
        f"Frame: "
        f"{st.session_state.replay_index}"
    )

else:

    st.info("Replay stopped.")

st.markdown("---")

if anomaly_prediction == -1:
    st.markdown("""
    <div style="
        background-color:#8B0000;
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:28px;
        font-weight:bold;
        color:white;
        margin-bottom: 25px;
        animation: blinker 1.2s linear infinite;
    ">
        CRITICAL PROCESS ANOMALY DETECTED
    </div>

    <style>
    @keyframes blinker {
        50% { opacity: 0.35; }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# KEY PROCESS METRICS GRID
# ============================================================

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        "Predicted Fault",
        f"Fault {predicted_fault}" if anomaly_prediction == -1 else "No Fault Detected"
    )

with metric_col2:
    st.metric(
        "Prediction Confidence",
        f"{confidence * 100:.2f}%"
    )

with metric_col3:
    if anomaly_prediction == -1:
        st.error("CRITICAL ANOMALY")
    else:
        st.success("NORMAL OPERATION")

with metric_col4:
    st.metric(
        "Process Health Score",
        f"{health_score:.1f}%"
    )

with metric_col4:

    st.metric(
        "LSTM Sequence Prediction",
        f"Fault {lstm_prediction}"
    )

    st.metric(
    "LSTM Confidence",
    f"{lstm_confidence * 100:.2f}%"
)

st.markdown("---")

# ============================================================
# TAB CONTENT
# ============================================================

if selected == "Process Monitoring":
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Live PCA Projection Space")
        st.markdown("Continuous projection of 52 process variables onto pre-trained principal component space.")
        
        try:
            pca_bg = pd.read_csv("data/pca_background.csv")
            fig_pca = go.Figure()
            
            # Plot background states
            for fault_num in sorted(pca_bg["Fault"].unique()):
                fault_subset = pca_bg[pca_bg["Fault"] == fault_num]
                fig_pca.add_trace(go.Scatter(
                    x=fault_subset["PC1"],
                    y=fault_subset["PC2"],
                    mode="markers",
                    name=f"Fault {fault_num}",
                    marker=dict(size=4, opacity=0.3),
                    showlegend=True
                ))
            
            # Overlay current live state
            fig_pca.add_trace(go.Scatter(
                x=[live_pca[0]],
                y=[live_pca[1]],
                mode="markers+text",
                name="LIVE Process Point",
                text=["LIVE STATE"],
                textposition="top center",
                marker=dict(
                    color="red",
                    size=16,
                    symbol="star",
                    line=dict(color="white", width=2),
                    opacity=1.0
                ),
                showlegend=True
            ))
            
            fig_pca.update_layout(
                xaxis_title="Principal Component 1 (PC1)",
                yaxis_title="Principal Component 2 (PC2)",
                template="plotly_dark",
                height=450,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering PCA chart: {e}")
            
    with chart_col2:
        st.subheader("Rolling Sensor History")
        st.markdown("Temporal movement of four primary reactor/stream sensor variables (sliding window).")
        
        history_df = pd.DataFrame(st.session_state.history)
        if not history_df.empty:
            st.line_chart(history_df)
        else:
            st.info("Gathering temporal data stream...")

elif selected == "Fault Detection":
    
    col_gauge, col_prob = st.columns(2)
    
    with col_gauge:
        st.subheader("Confidence Indicator")
        
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Prediction Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "cyan"},
                'steps': [
                    {'range': [0, 50], 'color': "#1e222b"},
                    {'range': [50, 80], 'color': "#2e3440"},
                    {'range': [80, 100], 'color': "#3b4252"}
                ]
            }
        ))
        
        gauge.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(
            gauge,
            use_container_width=True
        )
        
    with col_prob:
        st.subheader("Fault Probabilities")
        
        prob_df = pd.DataFrame({
            "Fault": [f"F{c}" for c in classifier.classes_],
            "Probability": probabilities
        })
        
        st.bar_chart(
            prob_df.set_index("Fault")
        )

elif selected == "Anomaly Analysis":
    
    col_drift, col_importance = st.columns([1, 2])
    
    with col_drift:
        st.subheader("Process Drift Diagnosis")
        st.markdown("Mathematical deviation from baseline fault-free operation.")
        
        st.markdown(f"""
        <div style="
            background-color: #1c1f26;
            border: 2px solid {drift_color};
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
        ">
            <h3 style="color: white; margin: 0;">DRIFT STATUS</h3>
            <h1 style="color: {drift_color}; font-size: 36px; margin: 15px 0;">{drift_status}</h1>
            <p style="color: #8c8c8c; margin: 0;">Metric Value: <b>{drift_score:.3f}</b> standard deviations</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_importance:
        st.subheader("Feature Importance Panel")
        st.markdown("Top critical variables contributing to fault classification.")
        
        try:
            feat_imp = pd.read_csv("data/feature_importances.csv")
            top_imp = feat_imp.head(10)
            
            fig_imp = go.Figure(go.Bar(
                x=top_imp["Importance"],
                y=top_imp["Feature"],
                orientation='h',
                marker=dict(color='cyan')
            ))
            
            fig_imp.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=10, b=10),
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading feature importance: {e}")
            
    st.markdown("---")
    st.subheader("Live Sensor Variables (52 channels)")
    
    live_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": st.session_state.input_data
    })
    
    st.dataframe(
        live_df,
        use_container_width=True,
        height=400
    )

# ============================================================
# EARLY WARNING SYSTEM
# ============================================================

st.subheader("Predictive Maintenance Alert")

if early_warning:

    st.error(
        "Emerging fault pattern detected. "
        "Process behavior indicates rising abnormality."
    )

else:

    st.success(
        "Process behavior stable."
    )

warning_df = pd.DataFrame({
    "Confidence":
    st.session_state.fault_history
})

st.line_chart(warning_df)

st.markdown("---")

# ============================================================
# ROOT CAUSE ANALYSIS PANEL
# ============================================================

st.subheader("Root Cause Analysis")

rootcause_df = pd.DataFrame({
    "Sensor": top_deviations.index,
    "Deviation": top_deviations.values
})

st.dataframe(
    rootcause_df,
    use_container_width=True
)

st.bar_chart(
    rootcause_df.set_index("Sensor")
)

st.markdown("---")

# ============================================================
# SENSOR HEATMAP
# ============================================================

st.subheader(
    "Sensor Abnormality Heatmap"
)

heatmap_data = np.array(
    sensor_deviation_df["Deviation"]
).reshape(4, 13)

heatmap_fig = px.imshow(
    heatmap_data,
    aspect="auto",
    labels=dict(
        color="Deviation"
    )
)

st.plotly_chart(
    heatmap_fig,
    use_container_width=True
)

st.markdown("---")

# ============================================================
# LIVE FAULT EVOLUTION
# ============================================================

st.subheader("Live Fault Probability Evolution")

if len(st.session_state.probability_history) > 0:

    prob_df = pd.DataFrame(
        st.session_state.probability_history
    )

    prob_df.columns = [
        f"Fault {i}"
        for i in range(1, 21)
    ]

    st.line_chart(prob_df)

# ============================================================
# LIVE PROCESS SIMULATION TRIGGER & DRIFT
# ============================================================
# REPLAY ENGINE
# ============================================================

if st.session_state.replay_active:

    if (
        st.session_state.replay_index
        < len(fault_data)
    ):

        current_row = fault_data.iloc[
            st.session_state.replay_index
        ]

        st.session_state.input_data = (
            current_row[feature_names]
            .values
            .astype(float)
        )

        st.session_state.replay_index += 1

    else:

        st.session_state.replay_active = False
else:
    # Simulate sensor drift
    noise = np.random.normal(
        0,
        0.5,
        len(st.session_state.input_data)
    )

    st.session_state.input_data = (
        st.session_state.input_data + noise
    )

st.session_state.sequence_buffer.append(
    st.session_state.input_data.copy()
)
# Append data to rolling sensor history
st.session_state.history.append({
    "xmeas_1": st.session_state.input_data[0],
    "xmeas_2": st.session_state.input_data[1],
    "xmeas_3": st.session_state.input_data[2],
    "xmeas_4": st.session_state.input_data[3]
})

# Keep only recent values (max 50)
if len(st.session_state.history) > 50:
    st.session_state.history.pop(0)

# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(0.2)
st.rerun()