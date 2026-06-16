import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("dataset/spam.csv")

# Features
X = df["text"]

# Labels
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()

# Train
model.fit(X_train_vec, y_train)

# Evaluate
predictions = model.predict(X_test_vec)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(
    model,
    "models/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("Training Complete")