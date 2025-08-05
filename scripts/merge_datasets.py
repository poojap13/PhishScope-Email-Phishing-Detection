import pandas as pd
import os

# File paths
kaggle_path = "data/Phishing_Email.csv"
synthetic_path = "data/synthetic_emails.csv"

# Check files exist
if not os.path.exists(kaggle_path):
    raise FileNotFoundError(f"Kaggle dataset not found at {kaggle_path}")
if not os.path.exists(synthetic_path):
    raise FileNotFoundError(f"Synthetic dataset not found at {synthetic_path}")

# Load datasets
df_kaggle = pd.read_csv(kaggle_path)
df_synthetic = pd.read_csv(synthetic_path)

# Standardize column names
df_kaggle.columns = [c.lower().strip() for c in df_kaggle.columns]
df_synthetic.columns = [c.lower().strip() for c in df_synthetic.columns]

# Merge datasets
df_combined = pd.concat([df_kaggle, df_synthetic], ignore_index=True)

# Shuffle rows
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

# Save full dataset (ignored in GitHub)
df_combined.to_csv("data/raw_emails.csv", index=False)

# Save small sample for GitHub
df_combined.sample(20, random_state=42).to_csv("data/sample_emails.csv", index=False)

print(f"✅ Merged dataset saved: {df_combined.shape[0]} rows total")
print("📂 Full dataset: data/raw_emails.csv")
print("📂 Sample dataset: data/sample_emails.csv")
