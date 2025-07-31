import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re

# Load dataset
df = pd.read_csv("emotion_dataset.csv")

# Basic text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters, keep spaces
    return text

df['text'] = df['text'].apply(clean_text)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['text'])
y = df['emotion']

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
with open("moodpal_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("moodpal_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")