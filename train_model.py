import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import joblib

def run_diagnostics(df, X_cols):
    """
    Runs ANOVA and KS-tests on all features to determine if there is any
    statistically significant difference in their distributions across classes.
    """
    print("\n" + "="*60)
    print("         DATA DIAGNOSTICS & STATISTICAL SIGNIFICANCE")
    print("="*60)
    
    y = df["Fault_Type"]
    X = df[X_cols]
    
    # 1. Class Distribution
    print("Class Counts:")
    counts = y.value_counts().sort_index()
    for cls, count in counts.items():
        pct = (count / len(y)) * 100
        print(f"  Class {cls}: {count:3d} samples ({pct:.1f}%)")
    
    # 2. ANOVA F-test
    f_scores, p_values = f_classif(X, y)
    anova_df = pd.DataFrame({
        "Feature": X_cols,
        "ANOVA F-Score": f_scores,
        "ANOVA p-value": p_values
    })
    
    # 3. Kolmogorov-Smirnov (KS) test vs Class 0 (with Bonferroni Correction)
    alpha = 0.05
    n_features = len(X_cols)
    bonferroni_limit = alpha / n_features
    
    ks_significant_counts = {1: 0, 2: 0, 3: 0}
    
    for fault in [1, 2, 3]:
        s0 = df[df["Fault_Type"] == 0]
        sf = df[df["Fault_Type"] == fault]
        for col in X_cols:
            stat, p_val = ks_2samp(s0[col], sf[col])
            if p_val < bonferroni_limit:
                ks_significant_counts[fault] += 1
                
    print("\nANOVA Summary:")
    sig_anova = anova_df[anova_df["ANOVA p-value"] < 0.05]
    print(f"  Features with ANOVA p-value < 0.05: {len(sig_anova)} / {n_features}")
    
    print("\nKolmogorov-Smirnov (KS) Test vs Class 0 (Bonferroni Corrected alpha = {:.6f}):".format(bonferroni_limit))
    for fault, count in ks_significant_counts.items():
        print(f"  Class 0 vs Class {fault}: {count} / {n_features} features show significant differences")
        
    # Check if features are purely random noise
    if len(sig_anova) == 0 and sum(ks_significant_counts.values()) == 0:
        print("\n" + "!"*60)
        print(" CRITICAL WARNING: ZERO STATISTICAL DIFFERENCE IN FEATURES!")
        print("!"*60)
        print(" Every single feature has the EXACT SAME distribution across all fault classes.")
        print(" ANOVA and KS-tests found no statistically significant separation.")
        print(" ")
        print(" Mathematical Implications:")
        print(" 1. The features are highly likely to be pure random noise relative to the target.")
        print(" 2. Any complex model (e.g. deep trees, large neural nets, gradient boosting)")
        print("    will easily achieve 100% training accuracy by memorizing noise, but")
        print("    will generalize no better than random guessing or the majority baseline on a test set.")
        print(" 3. This strongly suggests a data-collection or generation pipeline bug.")
        print("!"*60 + "\n")
        return True
    return False

def main():
    # Load dataset
    data_path = "data/industrial_fault_dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    X_cols = [col for col in df.columns if col != "Fault_Type"]
    
    # Run statistical diagnostics
    is_pure_noise = run_diagnostics(df, X_cols)
    
    # Split dataset (stratify to preserve class imbalance in splits)
    X = df[X_cols]
    y = df["Fault_Type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- MODEL 1: Dummy Baseline Classifier ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_scaled, y_train)
    dummy_pred = dummy.predict(X_test_scaled)
    dummy_acc = accuracy_score(y_test, dummy_pred)
    
    print("="*60)
    print("         1. BASELINE CLASSIFIER (Predict Majority Class)")
    print("="*60)
    print(f"Test Accuracy: {dummy_acc:.4f}")
    print("\nClassification Report (Dummy Classifier):")
    print(classification_report(y_test, dummy_pred, zero_division=0))
    
    # --- MODEL 2: Highly Regularized Random Forest ---
    # Restricting depth and leaf size prevents the trees from splitting on training noise
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    
    rf_train_acc = rf.score(X_train_scaled, y_train)
    rf_test_acc = rf.score(X_test_scaled, y_test)
    rf_pred = rf.predict(X_test_scaled)
    rf_macro_f1 = f1_score(y_test, rf_pred, average="macro")
    
    print("\n" + "="*60)
    print("         2. REGULARIZED RANDOM FOREST (Balanced Weights)")
    print("="*60)
    print(f"Train Accuracy: {rf_train_acc:.4f}")
    print(f"Test Accuracy:  {rf_test_acc:.4f}")
    print(f"Test Macro F1:  {rf_macro_f1:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, rf_pred, zero_division=0))
    
    # --- MODEL 3: Regularized Logistic Regression ---
    lr = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    
    lr_train_acc = lr.score(X_train_scaled, y_train)
    lr_test_acc = lr.score(X_test_scaled, y_test)
    lr_pred = lr.predict(X_test_scaled)
    lr_macro_f1 = f1_score(y_test, lr_pred, average="macro")
    
    print("\n" + "="*60)
    print("         3. WEIGHTED LOGISTIC REGRESSION (L2 Regularized)")
    print("="*60)
    print(f"Train Accuracy: {lr_train_acc:.4f}")
    print(f"Test Accuracy:  {lr_test_acc:.4f}")
    print(f"Test Macro F1:  {lr_macro_f1:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, lr_pred, zero_division=0))
    
    # Select best model for generalization
    # If the data is noise, Dummy (majority class) is actually the safest generalizer
    # because trying to predict minority classes on noise will only lead to false alarms (low precision).
    # However, if the user explicitly wants to deploy a model that attempts to catch faults,
    # we select the Regularized Random Forest because it is less prone to extreme false alarms.
    
    best_model = rf
    model_name = "Regularized Random Forest"
    
    print("\n" + "="*60)
    print("         MODEL SUMMARY & RECOMMENDATIONS")
    print("="*60)
    print(f"Saving the best model: {model_name}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/fault_classifier.pkl")
    print("Model successfully saved to 'models/fault_classifier.pkl'!")
    
    if is_pure_noise:
        print("\nACTIONABLE ADVICE TO INCREASE ACCURACY:")
        print(" 1. Fix the Data Generation Pipeline: Verify that the rows of features were not shuffled ")
        print("    or generated using pure random noise independent of the 'Fault_Type' label.")
        print(" 2. Check the FFT extraction: The columns 'FFT_Temp_X' etc. do not correspond to the actual ")
        print("    Fourier transform of the 'Temperature' columns. Make sure the sliding window and ")
        print("    frequency aggregation are correctly aligned with row indices.")
        print(" 3. Collect Real Time-Series Data: If you have access to the raw sequential sensor measurements, ")
        print("    we can compute actual sequential/temporal features (lags, rolling averages, rolling variance) ")
        print("    or run temporal deep learning architectures (LSTMs or 1D-CNNs).")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()