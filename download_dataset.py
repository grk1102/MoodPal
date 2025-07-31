from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("emotion")
# Convert train split to DataFrame
df = pd.DataFrame(dataset['train'])
# Rename columns for clarity (label: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise)
df = df.rename(columns={'label': 'emotion'})
# Map numeric labels to emotion names
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
df['emotion'] = df['emotion'].map(emotion_map)
# Save as CSV
df.to_csv("emotion_dataset.csv", index=False)
print("Dataset saved as emotion_dataset.csv")