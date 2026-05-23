import pyreadr
import pandas as pd

# Load data
result = pyreadr.read_r("data/Tennessee Eastman Process/TEP_Faulty_Training.RData")

# IMPORTANT:
# Check actual object name
print(result.keys())

# Load dataframe
df = result["faulty_training"]

print("Original Shape:", df.shape)

# Sample 5000 rows from each fault
subset_list = []

for fault in sorted(df["faultNumber"].unique()):
    sampled = df[df["faultNumber"] == fault].sample(
        5000,
        random_state=42
    )
    subset_list.append(sampled)

# Combine all subsets
subset = pd.concat(subset_list)

# Reset index
subset = subset.reset_index(drop=True)

print("Subset Shape:", subset.shape)

# Verify fault column exists
print("\nColumns:")
print(subset.columns)

print("\nFault Counts:")
print(subset["faultNumber"].value_counts().sort_index())

# Save
subset.to_csv("data/tep_subset.csv.csv", index=False)

print("\nSubset saved successfully!")