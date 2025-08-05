# MoodPal

## **MoodPal README**

### **Project Overview**  
MoodPal is a mental health companion application built with Streamlit that helps users track and understand their emotions through a user-friendly dashboard. It uses natural language processing (NLP) and sentiment analysis to detect emotions from text input, offers personalized coping strategies, and includes gamification elements to encourage regular use. The app is designed for both students and professionals, providing a supportive tool for mental well-being.

### **Features**  
- **Emotion Detection**: Analyzes user input (text and emojis) to identify emotions such as sadness, joy, love, anger, fear, and surprise with intensity levels (mild/strong).  
- **Mood Journal**: Allows users to export a PDF log of their mood entries with timestamps.  
- **Gamification**: Tracks interaction streaks and awards badges (e.g., Mood Explorer at 3 interactions, Consistency Star at 5 days).  
- **Song Playback**: Plays calming or upbeat songs based on detected emotions (sadness and surprise currently supported).  
- **Mood Insights**: Provides a summary of dominant emotions and average mood scores, visualized with trend charts.  
- **Dark Mode**: Offers a toggleable dark theme for better readability and user comfort.  

### **Prerequisites**  
- Python 3.8 or higher  
- Required libraries: `pandas`, `scikit-learn`, `nltk`, `streamlit`, `plotly`, `reportlab`, `pygame`  
- Model files: `moodpal_model.pkl` and `moodpal_vectorizer.pkl` (generated via `train_model.py`)  
- Sound files: `sounds/calming_song1.mp3`, `sounds/calming_song2.mp3`, `sounds/upbeat_song1.mp3`, `sounds/upbeat_song2.mp3`  
- Optional: `moodpal_logo.png` and `icons/{emotion}.png` for branding and visuals  

### **Installation**  
1. Clone the repository or download the `moodpal.py` file.  
2. Install the required dependencies:  

   pip install pandas scikit-learn nltk streamlit plotly reportlab pygame  

3. Ensure the NLTK VADER lexicon is downloaded(Python):  

   import nltk  
   nltk.download('vader_lexicon')  
 
4. Place the model files (`moodpal_model.pkl`, `moodpal_vectorizer.pkl`) and sound files in the project directory.  
5. (Optional) Add `moodpal_logo.png` to the project directory and emotion-specific icons (`icons/`) for enhanced visuals.  

### **Usage**  
1. Run the application:  

   streamlit run moodpal.py  
  
2. Open your browser at `http://localhost:8501`.  
3. Select your context (student or professional) from the sidebar.  
4. Enter how you feel in the "Share Your Mood" section and click "Submit".  
5. Provide more details in the follow-up section and click "Continue".  
6. Explore the dashboard in Stage 2, including:  
   - Detected emotion and intensity  
   - Mood insights and trend charts  
   - Song playback options  
   - Gamification progress and badges  
   - Mood journal export  
7. Toggle "Dark Mode" for a different theme, or click "Start Over" to reset.  

### **Development Notes**  
- The app uses a pre-trained MultinomialNB model with TF-IDF vectorization for emotion prediction.  
- Emotion detection combines VADER sentiment scores with emoji-based adjustments.  
- The code handles session state to maintain user progress across interactions.  
- Ensure all file dependencies are present to avoid runtime errors.  
