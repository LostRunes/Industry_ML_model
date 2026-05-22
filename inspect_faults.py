import pyreadr
import pandas as pd

# Load data
result = pyreadr.read_r("data/Tennessee Eastman Process/TEP_Faulty_Training.RData")

# Access dataframe
df = result["faulty_training"]

print("\nDataset Shape:")
print(df.shape)

print("\nFault Counts:")
print(df["faultNumber"].value_counts().sort_index())

print("\nUnique Faults:")
print(sorted(df["faultNumber"].unique()))