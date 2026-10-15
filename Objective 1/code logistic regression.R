# --- Load CSV ---
df <- read.csv("C:/Users/muham/Desktop/tng run test.csv", encoding = "UTF-8")

# Tukar nama column supaya seragam (optional tapi elok untuk jangka panjang)
names(df) <- trimws(names(df))
names(df)[names(df) == "Predicted.Sentiment"] <- "Predicted_Sentiment"
names(df)[names(df) == "True.Sentiment"] <- "True_Sentiment"

# Convert jadi factor
df$Predicted_Sentiment <- factor(trimws(df$Predicted_Sentiment))
df$True_Sentiment <- factor(trimws(df$True_Sentiment))

# Samakan level antara dua column
all_levels <- sort(unique(c(levels(df$Predicted_Sentiment),
                            levels(df$True_Sentiment))))
df$Predicted_Sentiment <- factor(df$Predicted_Sentiment, levels = all_levels)
df$True_Sentiment <- factor(df$True_Sentiment, levels = all_levels)

# Buat confusion matrix
library(caret)
conf_matrix <- caret::confusionMatrix(
  data = df$Predicted_Sentiment,
  reference = df$True_Sentiment
)

# Papar result
print(conf_matrix)


