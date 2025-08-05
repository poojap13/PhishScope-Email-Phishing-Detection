import streamlit as st
import pandas as pd
import joblib
import os

# ========================
# Load Model & Vectorizer
# ========================
MODEL_PATH = "models/phishscope_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="PhishScope - Phishing Email Detection", layout="wide")

st.title("📧 PhishScope - Phishing Email Detection App")
st.write("Upload a CSV file containing email **subject** and **body** to detect potential phishing emails.")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Ensure required columns exist
        if not {"subject", "body"}.issubset(df.columns):
            st.error("CSV must contain 'subject' and 'body' columns.")
        else:
            # Combine subject + body
            df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

            # Transform text
            X_tfidf = vectorizer.transform(df["text"])

            # Predictions
            df["phish_probability"] = model.predict_proba(X_tfidf)[:, 1]
            df["prediction"] = (df["phish_probability"] >= 0.5).astype(int)

            # Display results
            st.subheader("📊 Prediction Results")
            st.dataframe(df[["subject", "body", "prediction", "phish_probability"]].head(20))

            # Summary metrics
            total_phish = df["prediction"].sum()
            total_legit = len(df) - total_phish
            st.write(f"**Detected Phishing Emails:** {total_phish}")
            st.write(f"**Detected Legitimate Emails:** {total_legit}")

            # Download results
            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_download,
                file_name="phishscope_predictions.csv",
                mime="text/csv"
            )

            # Show top suspicious
            st.subheader("⚠️ Top 5 Most Suspicious Emails")
            top_suspicious = df.sort_values("phish_probability", ascending=False).head(5)
            st.table(top_suspicious[["subject", "phish_probability"]])
    except Exception as e:
        st.error(f"Error processing file: {e}")
