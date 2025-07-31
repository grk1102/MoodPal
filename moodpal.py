import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import pickle
import re

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Load model and vectorizer
with open(r"moodpal_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(r"moodpal_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit Dashboard
st.set_page_config(page_title="MoodPal", page_icon="😊")
st.title("MoodPal: From Words to Wellness, Your Mood, Your Pal")
st.markdown("Enter your thoughts, and let MoodPal support your mental well-being.")

# Context selection
context = st.selectbox("Select your context:", ["student", "professional"], key="context")

# Text input
user_input = st.text_area("How are you feeling today? (Emojis welcome!)", key="initial_input")
if st.button("Submit"):
    if user_input:
        # Clean text (same as training)
        cleaned_input = re.sub(r'[^a-z\s]', '', user_input.lower())
        # Vectorize input
        input_vec = vectorizer.transform([cleaned_input])
        # Predict emotion
        emotion = model.predict(input_vec)[0]
        # Get sentiment intensity
        scores = sid.polarity_scores(user_input)
        intensity = "mild" if abs(scores['compound']) < 0.5 else "strong"
        st.markdown(f"**Detected Emotion**: {emotion} ({intensity})")