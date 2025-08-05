import pandas as pd
import os

# Paths
raw_path = "data/raw_emails.csv"
cleaned_path = "data/cleaned_emails.csv"

# Check if file exists
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

# Load data
df = pd.read_csv(raw_path)

# ========================
# Cleaning
# ========================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop rows missing subject or body
df.dropna(subset=["subject", "body"], inplace=True)

# ========================
# Feature Engineering
# ========================

# Extract sender domain (if exists)
if "sender" in df.columns:
    df["sender_domain"] = df["sender"].apply(
        lambda x: x.split("@")[-1] if pd.notnull(x) else ""
    )
else:
    df["sender_domain"] = ""

# Count number of links in the 'links' column (if exists)
if "links" in df.columns:
    df["num_links"] = df["links"].apply(
        lambda x: len(str(x).split(",")) if pd.notnull(x) and str(x).strip() != "" else 0
    )
else:
    df["num_links"] = 0

# Suspicious keywords in subject/body
keywords = ["verify", "urgent", "password", "account", "click", "security", "login"]
df["suspicious_keyword_count"] = (
    df["subject"].str.lower().apply(lambda text: sum(kw in text for kw in keywords))
    + df["body"].str.lower().apply(lambda text: sum(kw in text for kw in keywords))
)

# ========================
# Save Cleaned Data
# ========================
os.makedirs("data", exist_ok=True)
df.to_csv(cleaned_path, index=False)

print(f"✅ Cleaned dataset saved at {cleaned_path} with shape: {df.shape}")
