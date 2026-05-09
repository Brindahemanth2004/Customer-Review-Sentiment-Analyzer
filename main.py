import pandas as pd
import os

# ---------------- LOAD DATA ----------------

df = pd.read_csv('data/reviews.csv')

print("Original Shape:", df.shape)

# ---------------- REMOVE NULLS ----------------

df = df.dropna(subset=['Review'])

# ---------------- RENAME COLUMNS ----------------

df.rename(columns={
    'Review': 'review',
    'Sentiment': 'sentiment'
}, inplace=True)

# ---------------- PREPROCESSING ----------------

from utils.preprocessing import clean_text

df['cleaned'] = df['review'].apply(clean_text)

print("\nSample Cleaned Data:")
print(df[['review', 'cleaned']].head())

# ---------------- BALANCE DATASET ----------------

print("\nBefore Balancing:")
print(df['sentiment'].value_counts())

min_count = df['sentiment'].value_counts().min()

df = (
    df.groupby('sentiment', group_keys=False)
    .apply(lambda x: x.sample(min_count, random_state=42))
    .reset_index(drop=True)
)

print("\nAfter Balancing:")
print(df['sentiment'].value_counts())

# ---------------- TF-IDF ----------------

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df['cleaned'])

y = df['sentiment']

print("\nTF-IDF Shape:", X.shape)

# ---------------- TRAIN TEST SPLIT ----------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL ----------------

from sklearn.svm import LinearSVC

model = LinearSVC(class_weight='balanced')

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ---------------- EVALUATION ----------------

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE MODEL ----------------

import joblib

os.makedirs('model', exist_ok=True)

joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("\nModel and Vectorizer Saved!")

# ---------------- TEST ----------------

test_reviews = [
    "This product is amazing",
    "Worst item ever",
    "It is okay",
    "Average quality",
    "Not bad not good",
    "The camera is good but battery is terrible",
    "Could have been better"
]

print("\nSample Predictions:\n")

for review in test_reviews:

    cleaned = clean_text(review)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    print(f"Review: {review}")
    print(f"Prediction: {prediction}")
    print("-" * 50)