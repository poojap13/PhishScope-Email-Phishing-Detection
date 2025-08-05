import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os

# ========================
# Load dataset
# ========================
df = pd.read_csv("data/cleaned_emails.csv")

# Combine subject + body
df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)

# ========================
# Noise injection function
# ========================
def add_noise(text):
    noise_words = ["update", "system", "hello", "check", "status", "please"]
    words = text.split()
    # Randomly insert noise words
    for _ in range(random.randint(1,3)):
        words.insert(random.randint(0, len(words)), random.choice(noise_words))
    return " ".join(words)

# Make 20% of legit emails look suspicious
legit_idx = df[df["label"] == 0].sample(frac=0.2, random_state=42).index
df.loc[legit_idx, "text"] = df.loc[legit_idx, "text"].apply(
    lambda x: x + " " + random.choice([
        "verify your account", "urgent login", "security update"
    ])
)

# Make 20% of phishing emails look more legit
phish_idx = df[df["label"] == 1].sample(frac=0.2, random_state=42).index
df.loc[phish_idx, "text"] = df.loc[phish_idx, "text"].apply(add_noise)

# ========================
# Train/Test Split
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ========================
# TF-IDF Vectorization (controlled complexity)
# ========================
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85
)

X_train_tfidf = vectorizer.fit_transform(X_train)

# ========================
# Logistic Regression Model
# ========================
model = LogisticRegression(max_iter=1000)

# Cross-validation on training set
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="accuracy")
print(f"Cross-validated Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full training set
model.fit(X_train_tfidf, y_train)

# ========================
# Evaluate on Holdout Test Set
# ========================
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

print("\nFinal Holdout Set Performance:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# ========================
# Save Model + Vectorizer
# ========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/phishscope_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved in 'models/' folder")
