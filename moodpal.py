import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import pickle
import emoji
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pygame import mixer
import textwrap
import os
import re
from datetime import datetime

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Initialize pygame mixer
mixer.init()

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
@st.cache_resource
def load_model():
    with open("moodpal_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_vectorizer():
    with open("moodpal_vectorizer.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    vectorizer = load_vectorizer()
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please run train_model.py first.")
    st.stop()

# Cache VADER analysis
@st.cache_data
def get_vader_scores(text):
    return sid.polarity_scores(text)

# Export mood journal as PDF
def export_mood_journal():
    pdf_file = "mood_journal.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(100, y, "MoodPal Journal")
    y -= 30
    for i, (input_text, emotion, timestamp) in enumerate(zip(st.session_state.inputs, st.session_state.emotions, st.session_state.timestamps), 1):
        text = f"Interaction {i} ({timestamp}): {input_text} -> {emotion.capitalize()}"
        for line in textwrap.wrap(text, width=80):
            c.drawString(100, y, line)
            y -= 20
            if y < 50:
                c.showPage()
                y = 750
    c.save()
    with open(pdf_file, "rb") as f:
        st.download_button("Download Mood Journal", f, file_name=pdf_file, key="download_journal", type="primary")

# Gamification: Update streak based on timestamps
def update_streak():
    if not st.session_state.get('last_interaction'):
        st.session_state.last_interaction = datetime.now().strftime("%Y-%m-%d")
        st.session_state.streak = 1
        return st.session_state.streak
    last_date = datetime.strptime(st.session_state.last_interaction, "%Y-%m-%d")
    current_date = datetime.now()
    delta = (current_date - last_date).days
    if delta == 1:
        st.session_state.streak = st.session_state.get('streak', 0) + 1
    elif delta > 1:
        st.session_state.streak = 1
    st.session_state.last_interaction = current_date.strftime("%Y-%m-%d")
    return st.session_state.streak

# Mood Insights Summary
def get_mood_insights():
    if not st.session_state.emotions or not st.session_state.mood_scores:
        return "No data yet. Share your mood to see insights!"
    df = pd.DataFrame({
        "Emotion": st.session_state.emotions,
        "Mood Score": st.session_state.mood_scores
    })
    dominant_emotion = df["Emotion"].mode()[0]
    avg_mood_score = df["Mood Score"].mean()
    mood_desc = "positive" if avg_mood_score > 0 else "negative" if avg_mood_score < 0 else "neutral"
    return f"Dominant Emotion: {dominant_emotion.capitalize()}<br>Average Mood: {mood_desc} (Score: {avg_mood_score:.2f})"

# Custom CSS for premium UI
def get_css(dark_mode=False):
    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        * {{
            box-sizing: border-box;
        }}
        .stApp {{
            background: {'linear-gradient(135deg, #E8F5E9 0%, #F5F6F5 100%)' if not dark_mode else 'linear-gradient(135deg, #1C2526 0%, #2E1B4F 100%)'};
            font-family: 'Inter', sans-serif;
            overflow: hidden;
        }}
        .sidebar .sidebar-content {{
            background: {'linear-gradient(180deg, #E8F5E9 0%, #F5F6F5 100%)' if not dark_mode else 'linear-gradient(180deg, #1C2526 0%, #2E1B4F 100%)'};
            color: {'#1A1A1A' if not dark_mode else '#E6ECEF'};
            padding: 15px;
            border-right: 2px solid {'#1976D2' if not dark_mode else '#4FC3F7'};
            border-radius: 0 8px 8px 0;
            box-shadow: 2px 0 4px rgba(0,0,0,0.1);
        }}
        .sidebar .sidebar-content h3 {{
            color: {'#1976D2' if not dark_mode else '#4FC3F7'};
            font-size: 26px;
            text-align: center;
            margin: 0 0 10px 0;
            text-transform: uppercase;
        }}
        .sidebar .sidebar-content a {{
            color: {'#1976D2' if not dark_mode else '#4FC3F7'};
            text-decoration: none;
            display: block;
            padding: 8px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }}
        .sidebar .sidebar-content a:hover {{
            background-color: {'rgba(25, 118, 210, 0.1)' if not dark_mode else 'rgba(79, 195, 247, 0.1)'};
            box-shadow: 0 0 5px {'rgba(25, 118, 210, 0.5)' if not dark_mode else 'rgba(79, 195, 247, 0.5)'};
        }}
        .main-container {{
            background-color: {'rgba(255, 255, 255, 0.85)' if not dark_mode else 'rgba(30, 30, 30, 0.85)'};
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            max-width: 1000px;
            margin: 0 auto;
            animation: fadeIn 1s ease-in;
            overflow: hidden;
        }}
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(15px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        .hero-section {{
            text-align: center;
            padding: 30px 0;
            margin: 0;
            animation: wave 2s infinite ease-in-out;
        }}
        @keyframes wave {{
            0% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-5px); }}
            100% {{ transform: translateY(0); }}
        }}
        .hero-title {{
            color: {'#1A1A1A' if not dark_mode else '#E6ECEF'};
            font-size: 48px;
            font-weight: 700;
            margin: 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        .hero-subtitle {{
            color: {'#1976D2' if not dark_mode else '#4FC3F7'};
            font-size: 24px;
            font-weight: 500;
            text-transform: uppercase;
            margin: 5px 0;
        }}
        
        .metric-card {{
            background: {'linear-gradient(45deg, #1976D2, #42A5F5)' if not dark_mode else 'linear-gradient(45deg, #4FC3F7, #0288D1)'};
            color: white;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            margin: 0;
            animation: popIn 0.5s ease-in;
        }}
        @keyframes popIn {{
            0% {{ transform: scale(0.95); opacity: 0; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        .emotion-card {{
            background-color: {'rgba(232, 245, 233, 0.9)' if not dark_mode else 'rgba(55, 71, 79, 0.9)'};
            padding: 15px;
            border-radius: 8px;
            border: 1px solid {'linear-gradient(45deg, #1976D2, #42A5F5)' if not dark_mode else 'linear-gradient(45deg, #4FC3F7, #0288D1)'};
            margin: 0;
            transition: transform 0.3s ease;
        }}
        .emotion-card:hover {{
            transform: scale(1.02);
        }}
        .emotion-text {{
            color: {'#1A1A1A' if not dark_mode else '#E6ECEF'};
            font-size: 26px;
            font-weight: 500;
            margin: 5px 0;
        }}
        .suggestion-text {{
            color: {'#F06292' if not dark_mode else '#FF8A80'};
            font-size: 22px;
            font-weight: 500;
            margin: 5px 0;
        }}
        .mood-indicator {{
            font-size: 48px;
            text-align: center;
            margin: 5px 0;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); }}
        }}
        .history-panel {{
            background-color: {'rgba(232, 245, 233, 0.9)' if not dark_mode else 'rgba(55, 71, 79, 0.9)'};
            padding: 15px;
            border-radius: 8px;
            margin: 0;
            border: 1px solid {'#E0E0E0' if not dark_mode else '#424242'};
        }}
        .history-item:nth-child(even) {{
            background-color: {'rgba(187, 222, 251, 0.5)' if not dark_mode else 'rgba(69, 90, 100, 0.5)'};
            padding: 6px;
            border-radius: 6px;
        }}
        .stButton>button {{
            background: {'linear-gradient(45deg, #1976D2, #42A5F5)' if not dark_mode else 'linear-gradient(45deg, #4FC3F7, #0288D1)'};
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background: {'linear-gradient(45deg, #1565C0, #1976D2)' if not dark_mode else 'linear-gradient(45deg, #0288D1, #4FC3F7)'};
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: scale(1.05);
        }}
        .suggestion-button:hover {{
            background: {'linear-gradient(45deg, #1565C0, #1976D2)' if not dark_mode else 'linear-gradient(45deg, #0288D1, #4FC3F7)'} !important;
            transform: scale(1.05);
        }}
        .stTextArea>label, .stSelectbox>label {{
            font-size: 18px;
            color: {'#1A1A1A' if not dark_mode else '#E6ECEF'};
            font-weight: 500;
            margin-bottom: 5px;
        }}
        .badge {{
            color: {'#F06292' if not dark_mode else '#FF8A80'};
            font-size: 18px;
            font-weight: 500;
            padding: 10px;
            background-color: {'rgba(255, 255, 255, 0.95)' if not dark_mode else 'rgba(50, 50, 50, 0.95)'};
            border-radius: 8px;
            text-align: center;
            margin: 5px 0;
            animation: bounceIn 0.6s ease-in;
            box-shadow: 0 0 8px rgba(240, 98, 146, 0.3);
        }}
        @keyframes bounceIn {{
            0% {{ transform: scale(0.8); opacity: 0; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        .progress-bar {{
            margin: 5px 0;
            font-size: 16px;
            color: {'#1976D2' if not dark_mode else '#4FC3F7'};
            font-weight: 500;
        }}
        .mini-chart {{
            background-color: {'rgba(232, 245, 233, 0.9)' if not dark_mode else 'rgba(55, 71, 79, 0.9)'};
            padding: 15px;
            border-radius: 8px;
            margin: 0;
            border: 1px solid {'#E0E0E0' if not dark_mode else '#424242'};
        }}
        .insights-card {{
            background-color: {'rgba(232, 245, 233, 0.9)' if not dark_mode else 'rgba(55, 71, 79, 0.9)'};
            padding: 15px;
            border-radius: 8px;
            border: 1px solid {'#1976D2' if not dark_mode else '#4FC3F7'};
            margin: 0;
            font-size: 16px;
            transition: transform 0.3s ease;
        }}
        .insights-card:hover {{
            transform: scale(1.02);
        }}
        .st-expander {{
            background-color: {'rgba(255, 255, 255, 0.9)' if not dark_mode else 'rgba(40, 40, 40, 0.9)'};
            border-radius: 8px;
            border: 1px solid {'#E0E0E0' if not dark_mode else '#424242'};
            margin: 0;
            padding: 0;
            overflow: hidden;
        }}
        .st-expander summary {{
            background: {'linear-gradient(45deg, #1976D2, #42A5F5)' if not dark_mode else 'linear-gradient(45deg, #4FC3F7, #0288D1)'};
            color: white;
            padding: 10px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s ease;
        }}
        .st-expander summary:hover {{
            background: {'linear-gradient(45deg, #1565C0, #1976D2)' if not dark_mode else 'linear-gradient(45deg, #0288D1, #4FC3F7)'};
        }}
        .stSubheader {{
            color: {'#1976D2' if not dark_mode else '#4FC3F7'};
            font-size: 26px;
            font-weight: 500;
            text-transform: uppercase;
            margin: 5px 0;
        }}
        </style>
    """
    return css

# Streamlit Dashboard
st.set_page_config(page_title="MoodPal", page_icon="❤️", layout="wide")

# Sidebar
with st.sidebar:
    try:
        st.image("moodpal_logo.png", caption="MoodPal", use_container_width=True)
    except FileNotFoundError:
        st.markdown("**MoodPal**")
    st.markdown("<h3>MoodPal</h3>", unsafe_allow_html=True)
    st.write("From Words to Wellness, Your Mood, Your Pal!")
    st.write("Your Mental Health Companion")
    st.markdown(
        """
        **Features:**
        - Emotion Detection
        - Mood Journal
        - Gamification
        - Song Playback
        - Mood Insights
        - Dark Mode
        """,
        unsafe_allow_html=True
    )
    st.markdown("**Contexts:** Student, Professional")
    st.markdown("---")
    st.write("© 2025 | Designed with ❤️")
    st.write("| For You. For Us,Always |")

# Main container
with st.container():
    # Dark mode toggle
    dark_mode = st.checkbox("Dark Mode", key="dark_mode")
    st.markdown(get_css(dark_mode), unsafe_allow_html=True)
    
    # Hero section
    st.markdown(
        """
        <div class="hero-section">
            <h2 class="hero-title">Welcome to MoodPal</h2>
            <p class="hero-subtitle">Your Companion for Mental Well-Being</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mood input section
    st.markdown("<div class='stSubheader'>Share Your Mood</div>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
        st.session_state.interactions = 0
        st.session_state.texts = []
        st.session_state.mood_scores = []
        st.session_state.emotions = []
        st.session_state.inputs = []
        st.session_state.timestamps = []
        st.session_state.suggestion_index = 0
        st.session_state.streak = 0
        st.session_state.last_interaction = None

    # Keyword-based sadness detection
    sadness_keywords = ["down", "sad", "tough", "stress", "overwhelm", "fail", "struggle"]

    # Context selection
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    context = st.selectbox("Select your context:", ["student", "professional"], key="context")
    st.markdown("</div>", unsafe_allow_html=True)

    # Stage 0: Initial input
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    user_input = st.text_area("How are you feeling today? (Emojis welcome!)", key="initial_input")
    if st.button("Submit", key="submit_initial", type="primary"):
        if user_input.strip():
            st.session_state.texts.append(user_input)
            st.session_state.inputs.append(user_input[:50] + "..." if len(user_input) > 50 else user_input)
            st.session_state.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # Perform emotion prediction
            combined_text = user_input
            emojis = [c for c in combined_text if c in emoji.EMOJI_DATA]
            emoji_score = sum(emoji_sentiment.get(e, 0) for e in emojis) / (len(emojis) + 1)
            cleaned_input = re.sub(r'[^a-z\s]', '', combined_text.lower())
            input_vec = vectorizer.transform([cleaned_input])
            text_emotion = model.predict(input_vec)[0]
            vader_scores = get_vader_scores(combined_text)
            text_score = vader_scores['compound']
            final_score = 0.4 * text_score + 0.6 * emoji_score
            score_to_emotion = {
                (-1.0, -0.03): "sadness",
                (-0.03, -0.01): "fear",
                (-0.01, 0.01): "surprise",
                (0.01, 0.15): "love",
                (0.15, 1.0): "joy",
            }
            final_emotion = text_emotion
            if any(e in ["😢", "😞", "😣", "😓"] for e in emojis) and vader_scores['neg'] > 0.03:
                final_emotion = "sadness"
            elif any(keyword in combined_text.lower() for keyword in sadness_keywords):
                final_emotion = "sadness"
            else:
                for (low, high), emotion in score_to_emotion.items():
                    if low <= final_score <= high:
                        final_emotion = emotion
                        break
            st.session_state.mood_scores.append(final_score)
            st.session_state.emotions.append(final_emotion)
            st.session_state.stage = 1
            st.session_state.interactions += 1
            update_streak()
            st.success("Mood submitted!", icon="✅")
        else:
            st.error("Please enter some text or emojis to continue.", icon="❌")
    st.markdown("</div>", unsafe_allow_html=True)

    # Stage 1: Follow-up question
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    follow_up = st.text_area("Share more details:", key="follow_up")
    if st.button("Continue", key="submit_follow_up", type="primary"):
        if follow_up.strip():
            st.session_state.texts.append(follow_up)
            st.session_state.inputs.append(follow_up[:50] + "..." if len(follow_up) > 50 else follow_up)
            st.session_state.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # Perform emotion prediction with weighted follow-up
            combined_text = f"{st.session_state.texts[0]} {follow_up} {follow_up}"
            emojis = [c for c in combined_text if c in emoji.EMOJI_DATA]
            emoji_score = sum(emoji_sentiment.get(e, 0) for e in emojis) / (len(emojis) + 1)
            cleaned_input = re.sub(r'[^a-z\s]', '', combined_text.lower())
            input_vec = vectorizer.transform([cleaned_input])
            text_emotion = model.predict(input_vec)[0]
            vader_scores = get_vader_scores(combined_text)
            text_score = vader_scores['compound']
            final_score = 0.4 * text_score + 0.6 * emoji_score
            score_to_emotion = {
                (-1.0, -0.03): "sadness",
                (-0.03, -0.01): "fear",
                (-0.01, 0.01): "surprise",
                (0.01, 0.15): "love",
                (0.15, 1.0): "joy",
            }
            final_emotion = text_emotion
            if any(e in ["😢", "😞", "😣", "😓"] for e in emojis) and vader_scores['neg'] > 0.03:
                final_emotion = "sadness"
            elif any(keyword in combined_text.lower() for keyword in sadness_keywords):
                final_emotion = "sadness"
            else:
                for (low, high), emotion in score_to_emotion.items():
                    if low <= final_score <= high:
                        final_emotion = emotion
                        break
            st.session_state.mood_scores.append(final_score)
            st.session_state.emotions.append(final_emotion)
            st.session_state.stage = 2
            st.session_state.interactions += 1
            update_streak()
            st.success("Details submitted!", icon="✅")
        else:
            st.error("Please provide more details to continue.", icon="❌")
    st.markdown("</div>", unsafe_allow_html=True)

    # Stage 2: Prediction and suggestion
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    if not st.session_state.texts or len(st.session_state.texts) != len(st.session_state.inputs) or len(st.session_state.inputs) != len(st.session_state.timestamps):
        st.session_state.stage = 0
        st.session_state.texts = []
        st.session_state.mood_scores = []
        st.session_state.emotions = []
        st.session_state.inputs = []
        st.session_state.timestamps = []
    else:
        combined_text = f"{st.session_state.texts[0]} {st.session_state.texts[1]} {st.session_state.texts[1]}"
        emojis = [c for c in combined_text if c in emoji.EMOJI_DATA]
        emoji_score = sum(emoji_sentiment.get(e, 0) for e in emojis) / (len(emojis) + 1)
        cleaned_input = re.sub(r'[^a-z\s]', '', combined_text.lower())
        input_vec = vectorizer.transform([cleaned_input])
        text_emotion = model.predict(input_vec)[0]
        vader_scores = get_vader_scores(combined_text)
        text_score = vader_scores['compound']
        final_score = 0.4 * text_score + 0.6 * emoji_score
        score_to_emotion = {
            (-1.0, -0.03): "sadness",
            (-0.03, -0.01): "fear",
            (-0.01, 0.01): "surprise",
            (0.01, 0.15): "love",
            (0.15, 1.0): "joy",
        }
        final_emotion = text_emotion
        if any(e in ["😢", "😞", "😣", "😓"] for e in emojis) and vader_scores['neg'] > 0.03:
            final_emotion = "sadness"
        elif any(keyword in combined_text.lower() for keyword in sadness_keywords):
            final_emotion = "sadness"
        else:
            for (low, high), emotion in score_to_emotion.items():
                if low <= final_score <= high:
                    final_emotion = emotion
                    break
        intensity = "mild" if abs(final_score) < 0.4 else "strong"
        
        # Display mood indicator
        mood_emoji = {"sadness": "😢", "joy": "😊", "love": "😍", "anger": "😡", "fear": "😱", "surprise": "😮"}
        mood_color = {
            "sadness": "#1976D2",
            "joy": "#43A047",
            "love": "#F06292",
            "anger": "#C62828",
            "fear": "#37474F",
            "surprise": "#7B1FA2"
        }
        st.markdown(f'<div class="mood-indicator" style="color: {mood_color.get(final_emotion, "#1976D2")}">{mood_emoji.get(final_emotion, "😊")}</div>', unsafe_allow_html=True)
        
        # Display emotion with card
        st.markdown("<div class='emotion-card'>", unsafe_allow_html=True)
        emotion_icon = f"icons/{final_emotion}.png"
        if os.path.exists(emotion_icon):
            st.image(emotion_icon, width=60, caption=f"{final_emotion.capitalize()} ({intensity})")
        st.markdown(f'<div class="emotion-text">Detected Emotion: {final_emotion.capitalize()} ({intensity})</div>', unsafe_allow_html=True)
        if emojis:
            st.markdown(f"**Emojis Detected**: {emojis} (Score: {emoji_score:.2f})")
        
        # Emotion intensity progress bar
        intensity_value = min(abs(final_score), 1.0)
        st.markdown(f'<div class="progress-bar">Emotion Intensity: {int(intensity_value * 100)}%</div>', unsafe_allow_html=True)
        st.progress(intensity_value)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Mood insights summary
        st.markdown("<div class='insights-card'>", unsafe_allow_html=True)
        st.markdown(f"**Mood Insights**<br>{get_mood_insights()}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Metric cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card' style='background: {'linear-gradient(45deg, #1976D2, #42A5F5)' if not dark_mode else 'linear-gradient(45deg, #4FC3F7, #0288D1)'}'><h4>Interactions</h4><h2>{st.session_state.interactions}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card' style='background: {'linear-gradient(45deg, #43A047, #66BB6A)' if not dark_mode else 'linear-gradient(45deg, #66BB6A, #43A047)'}'><h4>Streak</h4><h2>{st.session_state.streak} days</h2></div>", unsafe_allow_html=True)
        with col3:
            dominant_emotion = pd.DataFrame({"Emotion": st.session_state.emotions}).mode()["Emotion"][0] if st.session_state.emotions else "None"
            dominant_color = mood_color.get(dominant_emotion.lower(), "#1976D2" if not dark_mode else "#4FC3F7")
            st.markdown(f"<div class='metric-card' style='background: linear-gradient(45deg, {dominant_color}, {'#42A5F5' if not dark_mode else '#0288D1'})'><h4>Dominant Mood</h4><h2>{dominant_emotion.capitalize()}</h2></div>", unsafe_allow_html=True)
        
        # Emotion-based song playback
        song_map = {
            "sadness": [
                {"name": "Calming Piano", "file": "sounds/calming_song1.mp3"},
                {"name": "Soft Guitar", "file": "sounds/calming_song2.mp3"}
            ],
            "surprise": [
                {"name": "Happy Clappy", "file": "sounds/upbeat_song1.mp3"},
                {"name": "Upbeat Pop", "file": "sounds/upbeat_song2.mp3"}
            ]
        }
        song_desc = {
            "sadness": "a calming song to lift your spirits",
            "surprise": "an upbeat song to match your excitement"
        }
        if final_emotion in song_map:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"Would you like to play {song_desc[final_emotion]}?")
            song_options = [song["name"] for song in song_map[final_emotion]]
            selected_song = st.selectbox("Choose a song:", song_options, key="song_selection")
            if st.button("Play Song", key="play_song", type="primary"):
                selected_file = next(song["file"] for song in song_map[final_emotion] if song["name"] == selected_song)
                try:
                    mixer.music.load(selected_file)
                    mixer.music.play()
                except Exception as e:
                    st.warning(f"Could not play song: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Suggestion carousel
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        suggestion_key = (final_emotion.lower(), context)
        suggestion_list = strategies.get(suggestion_key, [{"text": "Stay balanced!", "weight": 1.0}])
        suggestion_list = sorted(suggestion_list, key=lambda x: x["weight"], reverse=True)
        if 'suggestion_index' not in st.session_state:
            st.session_state.suggestion_index = 0
        suggestion = suggestion_list[st.session_state.suggestion_index]["text"]
        st.markdown(f'<div class="suggestion-text">Suggestion: {suggestion}</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Next Suggestion", key="next_suggestion", help="View another suggestion"):
                st.session_state.suggestion_index = (st.session_state.suggestion_index + 1) % len(suggestion_list)
        with col2:
            if st.button("👍 Helpful", key="helpful", help="Mark suggestion as helpful"):
                for s in suggestion_list:
                    if s["text"] == suggestion:
                        s["weight"] += 0.1
                st.success("Thanks for the feedback!", icon="✅")
        with col3:
            if st.button("👎 Not Helpful", key="not_helpful", help="Mark suggestion as not helpful"):
                for s in suggestion_list:
                    if s["text"] == suggestion:
                        s["weight"] -= 0.1
                st.warning("We'll try a better suggestion next time!")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Mini mood trend chart
        if st.session_state.mood_scores:
            with st.expander("Mood Trend Preview", expanded=True):
                st.markdown('<div class="mini-chart">', unsafe_allow_html=True)
                df = pd.DataFrame({
                    "Interaction": range(1, len(st.session_state.mood_scores) + 1),
                    "Mood Score": st.session_state.mood_scores,
                    "Emotion": st.session_state.emotions,
                    "Input": st.session_state.inputs,
                    "Timestamp": st.session_state.timestamps
                })
                emotion_colors = {
                    "sadness": "#1976D2",
                    "joy": "#43A047",
                    "love": "#F06292",
                    "anger": "#C62828",
                    "fear": "#37474F",
                    "surprise": "#7B1FA2"
                }
                fig = px.line(
                    df,
                    x="Interaction",
                    y="Mood Score",
                    text="Emotion",
                    labels={"Mood Score": "Mood Score (-1 to 1)"},
                    title="Mood Trend Preview",
                    template="plotly_white",
                    color="Emotion",
                    color_discrete_map=emotion_colors,
                    hover_data=["Input", "Timestamp"]
                )
                fig.update_traces(mode="lines+markers+text", textposition="top center", line_width=2, marker=dict(size=8))
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", size=12, color="#1A1A1A" if not dark_mode else "#E6ECEF"),
                    showlegend=True,
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Full mood trend chart
        st.markdown("<div class='stSubheader'>Mood Analysis</div>", unsafe_allow_html=True)
        with st.expander("View Full Mood Trend", expanded=False):
            if (len(st.session_state.mood_scores) == len(st.session_state.emotions) == 
                len(st.session_state.inputs) == len(st.session_state.timestamps)):
                with st.spinner("Generating mood trend..."):
                    df = pd.DataFrame({
                        "Interaction": range(1, len(st.session_state.mood_scores) + 1),
                        "Mood Score": st.session_state.mood_scores,
                        "Emotion": st.session_state.emotions,
                        "Input": st.session_state.inputs,
                        "Timestamp": st.session_state.timestamps
                    })
                    emotion_colors = {
                        "sadness": "#1976D2",
                        "joy": "#43A047",
                        "love": "#F06292",
                        "anger": "#C62828",
                        "fear": "#37474F",
                        "surprise": "#7B1FA2"
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
                        hover_data=["Input", "Timestamp"]
                    )
                    fig.update_traces(mode="lines+markers+text", textposition="top center", line_width=2.5, marker=dict(size=10))
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", size=14, color="#1A1A1A" if not dark_mode else "#E6ECEF"),
                        showlegend=True,
                        hovermode="x unified",
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Mood trend data is incomplete. Please complete the current interaction or start over.", icon="❌")
        
        # Mood journal export
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        export_mood_journal()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # History panel
        if st.session_state.inputs:
            st.markdown("<div class='stSubheader'>Mood History</div>", unsafe_allow_html=True)
            with st.expander("View Mood History", expanded=True):
                st.markdown('<div class="history-panel">', unsafe_allow_html=True)
                st.markdown("**Your Mood History**")
                for i, (input_text, emotion, timestamp) in enumerate(zip(st.session_state.inputs, st.session_state.emotions, st.session_state.timestamps), 1):
                    st.markdown(f'<div class="history-item">**Interaction {i}** ({timestamp}): {input_text} → {emotion.capitalize()}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Gamification
        st.markdown("<div class='stSubheader'>Your Progress</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        progress = min(st.session_state.interactions / 5, 1.0)
        st.markdown(f'<div class="progress-bar">Progress to Next Badge: {int(progress * 100)}%</div>', unsafe_allow_html=True)
        st.progress(progress)
        if st.session_state.interactions >= 3:
            st.markdown('<div class="badge">🎉 Mood Explorer Badge: You\'re great at sharing your feelings!</div>', unsafe_allow_html=True)
        if st.session_state.interactions >= 5:
            st.markdown('<div class="badge">🌟 Wellness Warrior Badge: Keep nurturing your mental health!</div>', unsafe_allow_html=True)
        if st.session_state.streak >= 5:
            st.markdown('<div class="badge">🏆 Consistency Star: Amazing dedication over 5 consecutive days!</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Start Over", key="start_over", type="primary"):
            st.session_state.stage = 0
            st.session_state.texts = []
            st.session_state.mood_scores = []
            st.session_state.emotions = []
            st.session_state.inputs = []
            st.session_state.timestamps = []
            st.session_state.suggestion_index = 0
            st.success("Started over!", icon="✅")

    st.markdown("</div>", unsafe_allow_html=True)