import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/tep_subset.csv")

print("Original Shape:", df.shape)

# ============================================================
# SORT TEMPORALLY
# ============================================================

df = df.sort_values(
    by=["simulationRun", "sample"]
)

# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_cols = [
    col for col in df.columns
    if col not in [
        "faultNumber",
        "simulationRun",
        "sample"
    ]
]

# ============================================================
# CREATE TEMPORAL FEATURES
# ============================================================

window_size = 5

for col in feature_cols:

    # Rolling Mean
    df[f"{col}_rollmean"] = (
        df.groupby("simulationRun")[col]
        .transform(
            lambda x: x.rolling(window_size).mean()
        )
    )

    # Rolling Std
    df[f"{col}_rollstd"] = (
        df.groupby("simulationRun")[col]
        .transform(
            lambda x: x.rolling(window_size).std()
        )
    )

    # Delta / Rate of Change
    df[f"{col}_delta"] = (
        df.groupby("simulationRun")[col]
        .transform(
            lambda x: x.diff()
        )
    )

# ============================================================
# REMOVE NaNs CREATED BY ROLLING
# ============================================================

df = df.dropna()

print("New Shape:", df.shape)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "tep_temporal_features.csv",
    index=False
)

print("Temporal feature dataset saved!")