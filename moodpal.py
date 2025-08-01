import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import pickle
import emoji
import plotly.express as px
import base64
import os
import re

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Emoji sentiment mapping
emoji_sentiment = {
    "😊": 0.5, "😄": 0.7, "😢": -0.5, "😞": -0.7, "😣": -0.6, "😊👍": 0.6,
    "😡": -0.8, "😍": 0.8, "😱": -0.4, "🥳": 0.7, "😓": -0.6
}

# Coping strategies with weights
strategies = {
    ("sadness", "student"): [
        {"text": "Take a study break or talk to a counselor.", "weight": 1.0},
        {"text": "Listen to calming music or journal your thoughts.", "weight": 0.9}
    ],
    ("sadness", "professional"): [
        {"text": "Try a short walk or mindfulness exercise.", "weight": 1.0},
        {"text": "Take a moment to breathe deeply.", "weight": 0.9}
    ],
    ("joy", "student"): [
        {"text": "Share your positivity with classmates!", "weight": 1.0},
        {"text": "Celebrate with a fun activity!", "weight": 0.9}
    ],
    ("joy", "professional"): [
        {"text": "Keep inspiring your team!", "weight": 1.0},
        {"text": "Spread your enthusiasm in a meeting!", "weight": 0.9}
    ],
    ("anger", "student"): [
        {"text": "Try deep breathing or a quick stretch.", "weight": 1.0},
        {"text": "Write down what’s upsetting you to let it go.", "weight": 0.9}
    ],
    ("anger", "professional"): [
        {"text": "Step away for a moment to cool off.", "weight": 1.0},
        {"text": "Practice a quick mindfulness exercise.", "weight": 0.9}
    ],
    ("love", "student"): [
        {"text": "Express your feelings with a friend!", "weight": 1.0},
        {"text": "Write a kind note to someone special.", "weight": 0.9}
    ],
    ("love", "professional"): [
        {"text": "Share your positivity at work!", "weight": 1.0},
        {"text": "Appreciate a colleague’s effort today.", "weight": 0.9}
    ],
    ("fear", "student"): [
        {"text": "Write down your worries to clear your mind.", "weight": 1.0},
        {"text": "Talk to a trusted friend about your fears.", "weight": 0.9}
    ],
    ("fear", "professional"): [
        {"text": "Focus on one task at a time.", "weight": 1.0},
        {"text": "Break your work into smaller steps.", "weight": 0.9}
    ],
    ("surprise", "student"): [
        {"text": "Embrace the moment and share your excitement!", "weight": 1.0},
        {"text": "Tell a friend about this unexpected moment!", "weight": 0.9}
    ],
    ("surprise", "professional"): [
        {"text": "Channel this energy into your work!", "weight": 1.0},
        {"text": "Use this moment to spark creativity!", "weight": 0.9}
    ]
}

# Load model and vectorizer
try:
    with open("moodpal_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("moodpal_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please run train_model.py first.")
    st.stop()

# Function to encode image for background
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Image {image_path} not found. Using default background.")
        return ""

# Set up custom CSS with animations
background_image = get_base64_image("background.jpg")
if background_image:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{background_image}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .main-container {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            animation: fadeIn 1s ease-in;
            max-width: 900px;
            margin: auto;
        }}
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        .title {{
            color: #2E7D32;
            font-family: 'Arial', sans-serif;
            font-size: 44px;
            text-align: center;
            margin-bottom: 10px;
            animation: slideIn 0.5s ease-in;
        }}
        @keyframes slideIn {{
            0% {{ transform: translateX(-20px); opacity: 0; }}
            100% {{ transform: translateX(0); opacity: 1; }}
        }}
        .subtitle {{
            color: #388E3C;
            font-family: 'Arial', sans-serif;
            font-size: 20px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .welcome-text {{
            color: #4CAF50;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .emotion-card {{
            background-color: #E8F5E9;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            border: 1px solid #4CAF50;
            animation: popIn 0.3s ease-in;
        }}
        @keyframes popIn {{
            0% {{ transform: scale(0.95); opacity: 0; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        .emotion-text {{
            color: #1E88E5;
            font-size: 26px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .suggestion-text {{
            color: #F57C00;
            font-size: 22px;
            margin-top: 10px;
            animation: fadeIn 0.5s ease-in;
        }}
        .mood-indicator {{
            font-size: 40px;
            text-align: center;
            margin-top: 10px;
            color: #4CAF50;
        }}
        .debug-text {{
            color: #6B7280;
            font-size: 14px;
            margin-top: 10px;
        }}
        .history-panel {{
            background-color: #F1F8E9;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border: 1px solid #4CAF50;
        }}
        .stButton>button {{
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #388E3C;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
        }}
        .stTextArea>label {{
            font-size: 18px;
            color: #2E7D32;
        }}
        .badge {{
            color: #D81B60;
            font-size: 18px;
            font-weight: bold;
            margin-top: 15px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            text-align: center;
            animation: bounceIn 0.5s ease-in;
        }}
        @keyframes bounceIn {{
            0% {{ transform: scale(0.8); opacity: 0; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        .progress-bar {{
            margin-top: 10px;
            font-size: 16px;
            color: #388E3C;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit Dashboard
st.set_page_config(page_title="MoodPal", page_icon="😊", layout="wide")

# Display logo
try:
    st.image("moodpal_logo.png", width=120, caption="MoodPal")
except FileNotFoundError:
    st.markdown("**MoodPal**")

# Main container
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="title">MoodPal: From Words to Wellness, Your Mood, Your Pal</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter your thoughts, and let MoodPal support your mental well-being.</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-text">Welcome! Share how you’re feeling, and we’ll provide personalized support.</div>', unsafe_allow_html=True)

    # Context selection
    context = st.selectbox("Select your context:", ["student", "professional"], key="context")

    # Debug toggle
    show_debug = st.checkbox("Show debug info", value=True)

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
        st.session_state.interactions = 0
        st.session_state.texts = []
        st.session_state.mood_scores = []
        st.session_state.emotions = []
        st.session_state.inputs = []

    # Stage 0: Initial input
    if st.session_state.stage == 0:
        user_input = st.text_area("How are you feeling today? (Emojis welcome!)", key="initial_input")
        if st.button("Submit"):
            if user_input.strip():
                st.session_state.texts.append(user_input)
                st.session_state.inputs.append(user_input[:50] + "..." if len(user_input) > 50 else user_input)
                st.session_state.stage = 1
                st.session_state.interactions += 1
            else:
                st.error("Please enter some text or emojis to continue.")

    # Stage 1: Follow-up question
    elif st.session_state.stage == 1:
        st.markdown("Thanks for sharing! Can you tell us more about why you feel this way?")
        follow_up = st.text_area("Share more details:", key="follow_up")
        if st.button("Continue"):
            if follow_up.strip():
                st.session_state.texts.append(follow_up)
                st.session_state.inputs.append(follow_up[:50] + "..." if len(follow_up) > 50 else follow_up)
                st.session_state.stage = 2
                st.session_state.interactions += 1
            else:
                st.error("Please provide more details to continue.")

    # Stage 2: Prediction and suggestion
    elif st.session_state.stage == 2:
        if not st.session_state.texts:
            st.error("No input data available. Please start over.")
            st.session_state.stage = 0
            st.session_state.texts = []
            st.session_state.mood_scores = []
            st.session_state.emotions = []
            st.session_state.inputs = []
        else:
            combined_text = " ".join(st.session_state.texts)
            # Emoji analysis
            emojis = [c for c in combined_text if c in emoji.EMOJI_DATA]
            emoji_score = sum(emoji_sentiment.get(e, 0) for e in emojis) / (len(emojis) + 1)
            # Text prediction
            cleaned_input = re.sub(r'[^a-z\s]', '', combined_text.lower())
            input_vec = vectorizer.transform([cleaned_input])
            text_emotion = model.predict(input_vec)[0]
            vader_scores = sid.polarity_scores(combined_text)
            text_score = vader_scores['compound']
            # Combine scores
            final_score = 0.4 * text_score + 0.6 * emoji_score
            # Map score to emotion
            score_to_emotion = {
                (-1.0, -0.1): "sadness",
                (-0.1, -0.05): "fear",
                (-0.05, 0.05): "surprise",
                (0.05, 0.15): "love",
                (0.15, 1.0): "joy",
            }
            final_emotion = text_emotion
            # Prioritize emoji-driven sadness
            if any(e in ["😢", "😞", "😣", "😓"] for e in emojis) and vader_scores['neg'] > 0.1:
                final_emotion = "sadness"
            else:
                for (low, high), emotion in score_to_emotion.items():
                    if low <= final_score <= high:
                        final_emotion = emotion
                        break
            intensity = "mild" if abs(final_score) < 0.4 else "strong"
            
            # Debug output
            if show_debug:
                st.markdown(
                    f'<div class="debug-text">Debug: Text Score = {text_score:.2f}, '
                    f'Emoji Score = {emoji_score:.2f}, Final Score = {final_score:.2f}, '
                    f'Text Emotion = {text_emotion}, VADER Neg = {vader_scores["neg"]:.2f}</div>',
                    unsafe_allow_html=True
                )
            
            # Store emotion and score
            st.session_state.emotions.append(final_emotion)
            st.session_state.mood_scores.append(final_score)
            
            # Display mood indicator
            mood_emoji = {"sadness": "😢", "joy": "😊", "love": "😍", "anger": "😡", "fear": "😱", "surprise": "😮"}
            mood_color = {"sadness": "#1E88E5", "joy": "#4CAF50", "love": "#D81B60", "anger": "#F57C00", "fear": "#6B7280", "surprise": "#9C27B0"}
            st.markdown(f'<div class="mood-indicator" style="color: {mood_color.get(final_emotion, "#4CAF50")}">{mood_emoji.get(final_emotion, "😊")}</div>', unsafe_allow_html=True)
            
            # Display emotion with card
            st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
            emotion_icon = f"icons/{final_emotion}.png"
            if os.path.exists(emotion_icon):
                st.image(emotion_icon, width=50, caption=f"{final_emotion.capitalize()} ({intensity})")
            st.markdown(f'<div class="emotion-text">Detected Emotion: {final_emotion.capitalize()} ({intensity})</div>', unsafe_allow_html=True)
            if emojis:
                st.markdown(f"**Emojis Detected**: {emojis} (Score: {emoji_score:.2f})")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Get suggestion (highest-weighted)
            suggestion_key = (final_emotion.lower(), context)
            suggestion_list = strategies.get(suggestion_key, [{"text": "Stay balanced!", "weight": 1.0}])
            suggestion = max(suggestion_list, key=lambda x: x["weight"])["text"]
            st.markdown(f'<div class="suggestion-text">Suggestion: {suggestion}</div>', unsafe_allow_html=True)
            
            # Feedback for adaptive suggestions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful", key="helpful"):
                    for s in suggestion_list:
                        if s["text"] == suggestion:
                            s["weight"] += 0.1
                    st.markdown("Thanks for the feedback!")
            with col2:
                if st.button("👎 Not Helpful", key="not_helpful"):
                    for s in suggestion_list:
                        if s["text"] == suggestion:
                            s["weight"] -= 0.1
                    st.markdown("We'll try a better suggestion next time!")
            
            # History panel
            if st.session_state.inputs:
                st.markdown('<div class="history-panel">', unsafe_allow_html=True)
                st.markdown("**Your Mood History**")
                for i, (input_text, emotion) in enumerate(zip(st.session_state.inputs, st.session_state.emotions), 1):
                    st.markdown(f"**Interaction {i}**: {input_text} → {emotion.capitalize()}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Mood trend chart
            if st.button("View Mood Trend"):
                df = pd.DataFrame({
                    "Interaction": range(1, len(st.session_state.mood_scores) + 1),
                    "Mood Score": st.session_state.mood_scores,
                    "Emotion": st.session_state.emotions,
                    "Input": st.session_state.inputs
                })
                emotion_colors = {
                    "sadness": "#1E88E5",
                    "joy": "#4CAF50",
                    "love": "#D81B60",
                    "anger": "#F57C00",
                    "fear": "#6B7280",
                    "surprise": "#9C27B0"
                }
                fig = px.line(
                    df,
                    x="Interaction",
                    y="Mood Score",
                    text="Emotion",
                    labels={"Mood Score": "Mood Score (-1 to 1)"},
                    title="Your Mood Trend",
                    template="plotly_white",
                    color="Emotion",
                    color_discrete_map=emotion_colors,
                    hover_data=["Input"]
                )
                fig.update_traces(mode="lines+markers+text", textposition="top center", line_width=2.5, marker=dict(size=10))
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=14, color="#2E7D32"),
                    showlegend=True,
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Gamification
            progress = min(st.session_state.interactions / 5, 1.0)
            st.markdown(f'<div class="progress-bar">Progress to Next Badge: {int(progress * 100)}%</div>', unsafe_allow_html=True)
            st.progress(progress)
            if st.session_state.interactions >= 3:
                st.markdown('<div class="badge">🎉 Mood Explorer Badge: You\'re great at sharing your feelings!</div>', unsafe_allow_html=True)
            if st.session_state.interactions >= 5:
                st.markdown('<div class="badge">🌟 Wellness Warrior Badge: Keep nurturing your mental health!</div>', unsafe_allow_html=True)
            
            if st.button("Start Over"):
                st.session_state.stage = 0
                st.session_state.texts = []
                st.session_state.inputs = []

    st.markdown('</div>', unsafe_allow_html=True)