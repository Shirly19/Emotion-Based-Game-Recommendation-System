import pandas as pd

# Load your CSV log
csv_file = "training_log.csv"  # replace with your CSV path
df = pd.read_csv(csv_file)

# Keep only the first 70 epochs
df_clean = df[df['epoch'] < 70]  # or <=69 depending on zero-indexing

# Save to a new CSV
df_clean.to_csv("training_log_clean.csv", index=False)
print("Cleaned CSV saved! Rows kept:", len(df_clean))
