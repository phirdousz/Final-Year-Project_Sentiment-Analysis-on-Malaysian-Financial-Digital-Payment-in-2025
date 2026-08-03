# ============================================================
# 🔹 FINAL — LDA TOPIC MODELING (kwsp Dataset)
# ============================================================

# ---- Load Libraries ----
library(tm)
library(topicmodels)
library(tidytext)
library(ggplot2)
library(dplyr)
library(slam)
library(RColorBrewer)
library(topicdoc)
library(wordcloud)
library(LDAvis)
library(servr)
library(jsonlite)
library(textstem)

# ============================================================
# 1. LOAD DATA
# ============================================================
df <- read.csv("C:/Users/muham/Desktop/FYP 1/Data/kwsp run test.csv", encoding = "UTF-8")
texts <- df$Translated

# ============================================================
# 2. STOPWORDS SETTINGS
# ============================================================
eng_stop <- stopwords("en")

malay_stop <- c(
  "yang","dan","di","ke","dari","untuk","dengan","itu","ini",
  "ada","tidak","boleh","akan","sudah","dalam","pada","tak",
  "atau","juga","sebagai","lebih","oleh","apabila","bila","kerana",
  "utk","yg","kat","org","nak","je","xpe","xpaham"
)

kwsp_brand_stop <- c(
  "kwsp","epf","iakaun","akaun","isinar","ilestari","icitra","epfepf"
)

kwsp_noise_stop <- c(
  # Conversational noise
  "reply","say","even","one","people","now","much","many","lot",
  
  # Encoding / technical artefacts
  "â","ï","¶","±","‡","€","¡",
  "muh4r2025","ab408420","cdlanwc9","annamh98"
)

# FINAL (conservative)
all_stopwords <- unique(c(
  eng_stop,
  malay_stop,
  kwsp_brand_stop,
  kwsp_noise_stop
))

# ============================================================
# 3. TEXT CLEANING & DTM
# ============================================================
corpus <- VCorpus(VectorSource(texts)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(content_transformer(lemmatize_strings)) %>%
  tm_map(removeWords, all_stopwords) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus)
dtm <- dtm[row_sums(dtm) > 0, ]
dtm <- dtm[, col_sums(dtm) > 2] # ✔ Remove rare terms
dtm <- dtm[row_sums(dtm) > 0, ] # Recheck docs after column filter
dtm <- removeSparseTerms(dtm, 0.995)
dtm <- dtm[row_sums(dtm) > 0, ]

cat("Total valid docs:", nrow(dtm), "\n")

# ============================================================
# 4. MODEL SELECTION - PERPLEXITY
# (Use this in thesis discussion: “Lower = Better”)
# ============================================================
k_values <- 2:10
perplexities <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  model <- LDA(dtm, k, control=list(seed=1234))
  perplexities[i] <- perplexity(model, dtm)
  cat("K =", k, "→ Perplexity:", perplexities[i], "\n")
  rm(model)
  gc()
}

ggplot(data.frame(k=k_values, perplexity=perplexities),
       aes(k, perplexity)) +
  geom_line() + geom_point(size=3) +
  labs(title="Perplexity vs Bilangan Topik (K)",
       x="Topik (K)", y="Perplexity") +
  theme_minimal()

# Choose K where curve stabilizes or lowest (VISUALLY PICK)

# ============================================================
# 5. MODEL SELECTION - COHERENCE
# (Use this in thesis: “Higher = Better”)
# ============================================================
coh_scores <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  k <- k_values[i]
  model <- LDA(dtm, k, control=list(seed=1234))
  coh <- topic_coherence(model, dtm, top_n_tokens = 20) #Using optimal Top N
  coh_scores[i] <- mean(coh, na.rm=TRUE)
  cat("K =", k, "→ Coherence:", coh_scores[i], "\n")
  rm(model)
  gc()
}


ggplot(data.frame(k=k_values, coherence=coh_scores),
       aes(k, coherence)) +
  geom_line() + geom_point(size=3) +
  labs(title="Koherens vs Bilangan Topik (K)",
       x="Topik (K)", y="Koherens") +
  theme_minimal()

# Pick results that show best balance

# ============================================================
# 6. FINAL MODEL (EVALUATE BETWEEN 2 TESTS)
# ============================================================
num_topics <- 5  # Based on evaluation above
lda_model <- LDA(dtm, k = num_topics, method = "Gibbs",
                 control = list(seed = 1234, burnin = 500, thin = 100, iter = 1500))
cat("\n FINAL SELECTED TOPICS =", num_topics, "\n")

# ============================================================
# 7. TOP WORDS — Final Top N words for thesis tables
# ============================================================
top_n <- 20  

topics <- tidy(lda_model, matrix="beta") %>%
  group_by(topic) %>%
  slice_max(beta, n=top_n) %>%
  arrange(topic, desc(beta))

write.csv(topics, "lda_top_terms_kwsp.csv", row.names = FALSE)
print(topics)

# ============================================================
# 7A. HITUNG BILANGAN PERKATAAN PER TOPIC (FREKUENSI)
# ============================================================

dtm_tidy <- tidy(dtm)

# Gabungkan 'topics' dengan DTM tidy untuk mendapatkan frekuensi (count)
topic_word_counts <- topics %>%
  inner_join(dtm_tidy, by = c("term" = "term"), relationship = "many-to-many") %>%

  group_by(topic, term, beta) %>%
  summarise(
    total_words_in_top_n = sum(count),
    top_terms = paste(unique(term), collapse = ", "),
    .groups = 'drop'
  ) %>%
  arrange(topic)

cat("\n Bilangan Perkataan Keseluruhan (Frekuensi) untuk Top 20 Terma per Topik:\n")
print(topic_word_counts)


# Eksport hasil pengiraan ke CSV
write.csv(topic_word_counts, "lda_topic_word_counts_kwsp.csv", row.names = FALSE)
cat("✔ File Topic Word Counts Exported\n")

# ============================================================
# 8. EXPORT RESULTS — For Analysis in Excel / Visualization
# ============================================================
post <- posterior(lda_model)
phi <- post$terms
theta <- as.data.frame(post$topics)

df_clean <- df[as.numeric(rownames(dtm)), ]

final_output <- data.frame(
  Sentiment = df_clean$Predicted.Sentiment,
  Text = df_clean$Translated,
  theta
)

write.csv(final_output, "lda_final_results_kwsp.csv", row.names = FALSE)
cat("✔ File Exported\n")

# ============================================================
# 9. WORD CLOUDS
# ============================================================
for (i in 1:num_topics) {
  png(paste0("topic_", i, "_wc_kwsp.png"), 1200, 800)
  wc <- subset(topics, topic == i)
  wordcloud(wc$term, wc$beta, max.words=top_n,
            colors=brewer.pal(8,"Dark2"))
  title(paste("Topic kwsp", i))
  dev.off()
}

# ============================================================
# 10. INTERACTIVE LDAvis
# ============================================================
json_lda <- createJSON(
  phi = phi, theta = as.matrix(theta),
  doc.length = row_sums(dtm),
  vocab = colnames(phi),
  term.frequency = col_sums(dtm)
)

serVis(json_lda, out.dir="lda_vis_kwsp", open.browser=TRUE)
cat("🎉 Done!\n")

dim(df)
dim(df_clean)


