import os
from fpdf import FPDF

class IndustrialReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(140, 155, 165)
            self.cell(0, 10, "Industrial AI Monitoring System - Technical Documentation Report", align="L", ln=0)
            self.cell(0, 10, f"Page {self.page_no()}", align="R", ln=1)
            self.set_draw_color(0, 255, 210)
            self.set_line_width(0.5)
            self.line(10, 20, 200, 20)
            self.ln(5)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.set_text_color(140, 155, 165)
            self.cell(0, 10, "CONFIDENTIAL - ENTERPRISE CONTROL ROOM TELEMETRY DEPLOYMENT", align="C")

def create_report():
    pdf = IndustrialReport()
    pdf.set_margins(15, 25, 15)
    pdf.set_auto_page_break(auto=True, margin=20)

    # ============================================================
    # COVER PAGE
    # ============================================================
    pdf.add_page()
    pdf.set_fill_color(6, 9, 19)
    pdf.rect(0, 0, 210, 297, "F")

    # Glowing Border Effect
    pdf.set_draw_color(0, 255, 210)
    pdf.set_line_width(1.5)
    pdf.rect(10, 10, 190, 277, "D")

    # Title Banner Glow
    pdf.set_fill_color(29, 38, 79)
    pdf.rect(20, 70, 170, 45, "F")

    # Title Text
    pdf.set_y(80)
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "INDUSTRIAL AI", align="C", ln=1)
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(0, 255, 210)
    pdf.cell(0, 10, "MONITORING SYSTEM", align="C", ln=1)

    # Subtitle
    pdf.set_y(130)
    pdf.set_font("helvetica", "", 12)
    pdf.set_text_color(140, 155, 165)
    pdf.cell(0, 10, "TECHNICAL DOCUMENTATION REPORT", align="C", ln=1)
    pdf.cell(0, 10, "Tennessee Eastman Process Telemetry", align="C", ln=1)

    # Meta Information
    pdf.set_y(230)
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 6, "AUTHOR: Antigravity AI Engineering", align="C", ln=1)
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(140, 155, 165)
    pdf.cell(0, 6, "DEPLOYMENT TARGET: Enterprise Control Room Panel", align="C", ln=1)
    pdf.cell(0, 6, "SYSTEM VERSION: 3.1.0-Premium", align="C", ln=1)

    # ============================================================
    # PROJECT OVERVIEW
    # ============================================================
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(29, 38, 79)
    pdf.cell(0, 10, "1. Project Overview & Core Abstract", ln=1)
    pdf.set_draw_color(29, 38, 79)
    pdf.set_line_width(1)
    pdf.line(15, 33, 200, 33)
    pdf.ln(5)

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    overview_text = (
        "The Industrial AI Monitoring System represents a state-of-the-art predictive telemetry console "
        "engineered for safety-critical plant environments. Using the classical Tennessee Eastman Process (TEP) "
        "benchmark chemical plant simulator, the project builds a multi-tiered diagnostic intelligence suite.\n\n"
        "Traditionally, refinery operators face highly complex, high-dimensional alarm states that are difficult "
        "to diagnose rapidly. Our system addresses this by integrating real-time Classification (predicting "
        "active fault indexes), Anomaly Detection (distinguishing standard noise from structural drifts), "
        "and Deep Learning LSTM Sequence Trajectory Predictors to forecast process failures early. All mathematical "
        "insights are rendered on a high-fidelity dark cybernetic telemetry panel running at a simulated 5 frames "
        "per second control loop."
    )
    pdf.multi_cell(0, 6, overview_text)
    pdf.ln(5)

    # ============================================================
    # MODEL PERFORMANCE SUMMARY
    # ============================================================
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(29, 38, 79)
    pdf.cell(0, 10, "2. Model Performance Summary", ln=1)
    pdf.line(15, 95, 200, 95)
    pdf.ln(5)

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    perf_intro = (
        "To maximize diagnostic accuracy, three distinct ML/DL model types were evaluated. Below is the "
        "comparative validation summary:"
    )
    pdf.multi_cell(0, 6, perf_intro)
    pdf.ln(5)

    # Performance Table
    pdf.set_font("helvetica", "B", 10)
    pdf.set_fill_color(29, 38, 79)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(90, 8, "Model Type / Architecture", border=1, fill=True, align="L")
    pdf.cell(90, 8, "Classification Accuracy", border=1, fill=True, align="R", ln=1)

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(90, 8, "Static Random Forest (Static RF)", border=1, align="L")
    pdf.cell(90, 8, "72.00%", border=1, align="R", ln=1)
    
    pdf.cell(90, 8, "Temporal Random Forest (Temporal RF)", border=1, align="L")
    pdf.cell(90, 8, "96.10%", border=1, align="R", ln=1)
    
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(90, 8, "Deep Sequential LSTM (Temporal LSTM)", border=1, align="L")
    pdf.cell(90, 8, "97.66%", border=1, align="R", ln=1)
    
    pdf.ln(5)
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "*Note: Static RF struggles to capture transitional states because it evaluates measurements independent of history. Temporal RF and LSTM models solve this by utilizing multi-frame sliding window buffers, bringing diagnostic rates to near-perfect levels.")
    pdf.ln(5)

    # ============================================================
    # WORKSPACE FILE STRUCTURE
    # ============================================================
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(29, 38, 79)
    pdf.cell(0, 10, "3. Project Workspace File Structure", ln=1)
    pdf.line(15, 33, 200, 33)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 6, "/ (Root Folder)", ln=1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    root_files = (
        " - download_dataset.py: Script that automatically downloads the base TEP subset (tep_subset.csv).\n"
        " - download_models.py: Script that automatically retrieves pre-trained network model binaries.\n"
        " - prepare_replay_data.py: Sorts the datasets temporally and prepares the 10,000-frame TEP replay data.\n"
        " - train_tep_model.py: Trains the multiclass Random Forest fault classifier.\n"
        " - train_anomaly_detector.py: Fits the Isolation Forest anomaly detector on baseline data.\n"
        " - train_lstm_model.py: Configures and trains the deep sequence LSTM neural network weights.\n"
        " - create_temporal_features.py: Extracts rolling mean, std, and delta parameters over window buffers.\n"
        " - train_temporal_model.py: Trains a Random Forest on extracted temporal features.\n"
        " - requirements.txt: Defines external dependencies (TensorFlow, scikit-learn, joblib, plotly, streamlit).\n"
        " - README.md: Comprehensive markdown configuration guide."
    )
    pdf.multi_cell(0, 6, root_files)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 6, "/app/ (User Interface)", ln=1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    app_files = (
        " - dashboard.py: The premium HMI control room dashboard built on Streamlit with deep Custom CSS styling."
    )
    pdf.multi_cell(0, 6, app_files)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 6, "/data/ (Telemetry Storage)", ln=1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    data_files = (
        " - tep_subset.csv: Base Tennessee Eastman dataset.\n"
        " - tep_replay.csv: Ordered fault replay file prepared by prepare_replay_data.py.\n"
        " - presets.json: Telemetry presets mapping to standard/abnormal states.\n"
        " - incident_logs.csv: Autonomous logging registry for recorded critical faults.\n"
        " - feature_importances.csv & pca_background.csv: Support data for visual rendering panels."
    )
    pdf.multi_cell(0, 6, data_files)
    pdf.ln(5)

    # ============================================================
    # ADVANCED TELEMETRY SYSTEMS
    # ============================================================
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(29, 38, 79)
    pdf.cell(0, 10, "4. Advanced Control Panel Systems", ln=1)
    pdf.line(15, 33, 200, 33)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 6, "A. Root Cause Analysis (RCA) Engine", ln=1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    rca_desc = (
        "To resolve the operational challenge of rapid fault localization, the system computes the exact "
        "deviation values between the live active sensor measurements and a cached normal operating baseline. "
        "It dynamically ranks these deviations and plots them to instantly highlight abnormal process variables."
    )
    pdf.multi_cell(0, 6, rca_desc)
    pdf.ln(3)

    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 6, "B. Spatial Telemetry Heatmap", ln=1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    heatmap_desc = (
        "The 52 process variables are projected into a physical 4 x 13 spatial control grid representation. "
        "Using Plotly Express turbo-scaling, the layout represents the actual plant flow. Crimson glows instantly "
        "pinpoint highly perturbed process sectors during pre-failure states."
    )
    pdf.multi_cell(0, 6, heatmap_desc)
    pdf.ln(3)

    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 6, "C. Predictive Warning Early Alert", ln=1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    warning_desc = (
        "Monitors rising confidence trajectories inside a sliding window. If a continuous positive trend is detected "
        "and the prediction exceeds a 60% probability threshold, the panel triggers an early predictive maintenance "
        "alert, allowing operators to halt plant processes before catastrophic failure."
    )
    pdf.multi_cell(0, 6, warning_desc)
    pdf.ln(3)

    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 6, "D. High-Speed HMI Replay Loop", ln=1)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    loop_desc = (
        "Features sidebar selection sliders and sequential replay engines. The UI autorefresh loop runs at "
        "5 frames per second server-side, bringing a high-speed telemetry experience identical to real-world HMIs."
    )
    pdf.multi_cell(0, 6, loop_desc)

    pdf.output("Industrial_AI_Monitoring_System_Report.pdf")
    print("Report generated successfully!")

if __name__ == "__main__":
    create_report()
