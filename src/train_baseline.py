# src/train_baseline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Load cleaned dataset
    df = pd.read_csv("data/liar2_clean.csv")
    X = df["clean"]
    y = df["label"]  # 6-class labels: 0–5

    # 2. Train/test split (80/20 with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Vectorize using TF-IDF (up to bigrams)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 4. Train Logistic Regression (max_iter for convergence)
    model = LogisticRegression(max_iter=1_000)
    model.fit(X_train_tfidf, y_train)

    # 5. Predict and evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
