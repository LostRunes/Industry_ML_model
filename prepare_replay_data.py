import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/tep_subset.csv")

# ============================================================
# SORT TEMPORALLY
# ============================================================

df = df.sort_values(
    by=["faultNumber", "simulationRun", "sample"]
)

# ============================================================
# SAVE SMALLER REPLAY DATASET
# ============================================================

# Keep only first 1 simulation run per fault
replay_df = (
    df.groupby("faultNumber")
      .head(500)
)

print("Replay Shape:")
print(replay_df.shape)

# ============================================================
# SAVE
# ============================================================

replay_df.to_csv(
    "data/tep_replay.csv",
    index=False
)

print("\nReplay dataset saved!")