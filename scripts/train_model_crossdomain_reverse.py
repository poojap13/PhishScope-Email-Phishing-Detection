import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# ========================
# Load datasets
# ========================
kaggle_df = pd.read_csv("data/Phishing_Email.csv")
synthetic_df = pd.read_csv("data/synthetic_emails.csv")

# Standardize column names
kaggle_df.columns = kaggle_df.columns.str.lower()
synthetic_df.columns = synthetic_df.columns.str.lower()

# Convert labels to numeric
kaggle_df["label"] = kaggle_df["email type"].apply(lambda x: 1 if "phishing" in x.lower() else 0)
synthetic_df["label"] = synthetic_df["label"].astype(int)

# Create unified text column
kaggle_df["text"] = kaggle_df["email text"].astype(str)
synthetic_df["text"] = synthetic_df["subject"].astype(str) + " " + synthetic_df["body"].astype(str)

# Split into train (Synthetic) and test (Kaggle)
X_train = synthetic_df["text"]
y_train = synthetic_df["label"]

X_test = kaggle_df["text"]
y_test = kaggle_df["label"]

# ========================
# TF-IDF Vectorization
# ========================
vectorizer = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========================
# Train Model
# ========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ========================
# Evaluate on Kaggle Test Set
# ========================
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

print("\n📊 Cross-Domain Evaluation (Train: Synthetic, Test: Kaggle)")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# ========================
# Save Model + Vectorizer
# ========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/phishscope_model_reverse.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer_reverse.pkl")

print("\n✅ Reverse model and vectorizer saved in 'models/' folder")
