import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("tep_subset.csv")

# Features
X = df.drop(columns=[
    "faultNumber",
    "simulationRun",
    "sample"
])

# Labels
y = df["faultNumber"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Most Important Features:\n")
print(importance_df.head(15))

# Plot
plt.figure(figsize=(12, 8))

plt.barh(
    importance_df["Feature"][:15][::-1],
    importance_df["Importance"][:15][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Important Features for TEP Fault Classification")

plt.tight_layout()

plt.show()