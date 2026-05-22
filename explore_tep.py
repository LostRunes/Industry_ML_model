import pyreadr

# Load one file
result = pyreadr.read_r("data/Tennessee Eastman Process/TEP_FaultFree_Training.RData")

# Show object names
print("\nObjects inside file:")
print(result.keys())

# Access first object
for key in result.keys():
    df = result[key]

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    break