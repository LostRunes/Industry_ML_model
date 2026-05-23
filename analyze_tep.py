import pandas as pd

# Load subset
df = pd.read_csv("data/tep_subset.csv.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nFault Counts:")
print(df["faultNumber"].value_counts().sort_index())

print("\nBasic Statistics:")
print(df.describe())