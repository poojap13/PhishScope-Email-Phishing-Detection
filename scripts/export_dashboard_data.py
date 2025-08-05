import pandas as pd
import joblib
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_emails.csv")

# Load model & vectorizer
model = joblib.load("models/phishscope_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Combine subject + body
df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

# Predict phishing probabilities
df["phish_probability"] = model.predict_proba(vectorizer.transform(df["text"]))[:, 1]

# OPTIONAL: Create a binary prediction (0 = legit, 1 = phishing)
df["predicted_label"] = (df["phish_probability"] >= 0.5).astype(int)

# Save for Power BI
os.makedirs("dashboard", exist_ok=True)
df.to_csv("dashboard/phishscope_dashboard_data.csv", index=False)
print("✅ Data exported for Power BI: dashboard/phishscope_dashboard_data.csv")
