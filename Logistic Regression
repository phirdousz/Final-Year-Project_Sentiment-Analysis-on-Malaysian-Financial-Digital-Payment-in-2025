# ============================================================
#  SENTIMENT ANALYSIS — Logistic Regression Model (Objektif 1)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# LOAD & CLEAN DATA
# ============================================================

df_path = r"C:\Users\muham\Desktop\FYP 1\Data\kwsp run test.csv"
# df_path = r"C:\Users\muham\Desktop\FYP 1\Data\asb run test.csv"
# df_path = r"C:\Users\muham\Desktop\FYP 1\Data\tng run test.csv"
df = pd.read_csv(df_path, encoding="utf-8-sig")

print("Columns in df:", df.columns.tolist())

# Buang nilai kosong dan kosong string
df = df.dropna(subset=["Translated"])
df = df[df["Translated"].str.strip() != ""]

# Pastikan label sentimen konsisten
df["True Sentiment"] = df["True Sentiment"].astype(str).str.strip().str.lower()
df = df[df["True Sentiment"].isin(["positive", "negative", "neutral"])]
df = df.reset_index(drop=True)

# ============================================================
# SPLIT DATA — Train/Test
# ============================================================

X = df["Translated"]
y = df["True Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# FEATURE EXTRACTION — TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ============================================================
# MODEL TRAINING
# ============================================================

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# ============================================================
# MODEL EVALUATION (TEST SET)
# ============================================================

y_pred = model.predict(X_test_vec)

print("\nClassification Report (TEST SET only):\n")
print(classification_report(y_test, y_pred, digits=2))

cm_test = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
print("\nConfusion Matrix (TEST SET):\n", cm_test)

# ============================================================
# APPLY MODEL TO FULL DATASET (UNTUK SIMPAN DALAM CSV)
# ============================================================

X_all_vec = vectorizer.transform(X)
all_pred = model.predict(X_all_vec)
all_pred_proba = model.predict_proba(X_all_vec)
sentiment_scores = all_pred_proba.max(axis=1)

# Simpan hasil ke dataset
df["Predicted Sentiment"] = all_pred
df["sentiment_score"] = sentiment_scores

df.to_csv(df_path, index=False, encoding="utf-8-sig")
print(f"\n Predictions and scores overwritten in: {df_path}")

# ============================================================
# TABURAN SENTIMEN (TRUE vs PREDICTED)
# ============================================================

print("\n Sentiment distribution (TRUE labels):")
print(y.value_counts())

print("\n Sentiment distribution (PREDICTED labels):")
print(pd.Series(all_pred).value_counts())

# ============================================================
# OVERALL SENTIMENT SCORE (Purata Keyakinan)
# ============================================================

overall_score = sentiment_scores.mean()
print(f"\n Average Sentiment Confidence Score: {overall_score:.2f}")

# ============================================================
# VISUALIZATION DASHBOARD
# ============================================================

report = classification_report(y_test, y_pred, output_dict=True)
cm_full = confusion_matrix(y, all_pred, labels=["positive", "neutral", "negative"])

fig, axes = plt.subplots(3, 2, figsize=(12, 14))

# Confusion Matrix (Test Set)
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Positif", "Neutral", "Negatif"],
            yticklabels=["Positif", "Neutral", "Negatif"],
            ax=axes[0,0])
axes[0,0].set_title("Matrix Kekeliruan (Set Latihan)")
axes[0,0].set_ylabel("Label Sebenar")
axes[0,0].set_xlabel("Label Jangkaan")

# Confusion Matrix (Full Dataset)
sns.heatmap(cm_full, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Positif", "Neutral", "Negatif"],
            yticklabels=["Positif", "Neutral", "Negatif"],
            ax=axes[0,1])
axes[0,1].set_title("Matrix Kekeliruan (Set Penuh)")
axes[0,1].set_ylabel("Label Sebenar")
axes[0,1].set_xlabel("Label Jangkaan")

# Performance Metrics (Precision, Recall, F1)
metrics_df = pd.DataFrame(report).transpose().iloc[:-3, :3]
metrics_df = metrics_df.rename(columns={
    "precision": "Kejituan", 
    "recall": "Ingat kembali", 
    "f1-score": "Skor f1"
})
metrics_df = metrics_df.rename(index={
    "positive": "Positif", 
    "negative": "Negatif", 
    "neutral": "Neutral"
})

metrics_df.plot(kind="bar", ax=axes[1,0], color=["blue", "darkorange", "green"])
axes[1,0].set_title("Prestasi Model")
axes[1,0].set_ylabel("Skor")
axes[1,0].set_ylim(0, 1)
axes[1,0].tick_params(axis='x', labelrotation=0)

axes[1,1].axis("off")

# True Sentiment Distribution
y.value_counts().reindex(["positive", "neutral", "negative"], fill_value=0).rename(
    index={'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Neutral'}
).plot(
    kind="pie", autopct="%1.1f%%", startangle=90, shadow=True,
    colors=["green", "orange", "red"], ax=axes[2,0] 
)
axes[2,0].set_ylabel("")
axes[2,0].set_title("Taburan Sentimen Sebenar")

# Predicted Sentiment Distribution
pd.Series(all_pred).value_counts().reindex(["positive", "neutral", "negative"], fill_value=0).rename(
    index={'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Neutral'}
).plot(
    kind="pie", autopct="%1.1f%%", startangle=90, shadow=True,
    colors=["green", "orange", "red"], ax=axes[2,1]
)
axes[2,1].set_ylabel("")
axes[2,1].set_title("Taburan Sentimen Jangkaan")

plt.suptitle("Papan Pemuka Analisis Sentimen KWSP", fontsize=16)
# plt.suptitle("Papan Pemuka Analisis Sentimen ASB", fontsize=16)
# plt.suptitle("Papan Pemuka Analisis Sentimen TNG", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
