#  SENTIMENT ANALYSIS — VADER Model (Objektif 1)

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

# ---------- Load dataset ----------
df_path = r"C:\Users\muham\Desktop\FYP 1\Data\kwsp run test.csv"
# df_path = r"C:\Users\muham\Desktop\FYP 1\Data\asb run test.csv"
# df_path = r"C:\Users\muham\Desktop\FYP 1\Data\tng run test.csv"
df = pd.read_csv(df_path, encoding="utf-8-sig")

print("Columns in df before:", df.columns.tolist())

# ---------- Initialize VADER ----------
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        sentiment = "positive"
    elif score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    # return both sentiment + score (string to avoid Excel issues)
    return pd.Series([sentiment, f"{score:.4f}"])  

# ---------- Generate Suggested Sentiment + Score ----------
df[["True Sentiment", "sentiment_score"]] = df["Translated"].apply(analyze_sentiment)

# ---------- Save back to CSV ----------
df.to_csv(df_path, index=False, encoding="utf-8-sig")

print("Suggested sentiments + scores saved into CSV")
print("\nSuggested Sentiment Distribution:")
print(df["True Sentiment"].value_counts())
