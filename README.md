# Sentiment Analysis - Identifying Toxic Comments 🚨💬

### Author: Arpita Lonakadi  
**Email:** lonakadiarpita@gmail.com
**Date:** September 29, 2024

---

## 📌 Project Overview

This project focuses on identifying toxic comments from social media using machine learning techniques and Natural Language Processing (NLP). Toxic comments are defined as rude, disrespectful, or inappropriate content that may disrupt healthy online discussions. The solution involves building and evaluating multiple models, integrating sentiment scores from emojis and VADER, and optimizing model performance using hyperparameter tuning and threshold adjustment.

---

## 🧠 Objective

To develop a scalable, accurate classification model that can:
- Automatically detect toxic comments on social media platforms.
- Improve recall (especially for toxic comments).
- Support future moderation pipelines for social networks.

---

## 📂 Dataset Description

### 🔹 Training Dataset
- Social media comments from platforms like Reddit, Twitter, and YouTube.
- Labeled as `toxic (1)` or `non-toxic (0)` using majority voting based on `composite_toxic` field.

### 🔹 Testing Dataset
- Similar unlabeled comments used to evaluate model performance.

### Key Columns:
- `text`, `platform_id`, `article_url`, `parent_comment`, `composite_toxic`, `platform`.

---

## 🧹 NLP Preprocessing Steps

To prepare the data for modeling, we applied several NLP techniques:

### 1. **Lowercasing**
All text was converted to lowercase to ensure consistency.

### 2. **Removing User Mentions**
Handles @user patterns, which don't influence sentiment directly.

### 3. **Removing URLs**
Cleans out links using regex to remove unrelated noise.

### 4. **Emoji Handling**
- Used `emoji` Python library to convert emojis into their textual descriptions.
- Assigned **emoji sentiment scores**:  
  - `+1` for positive (e.g., 😊, 😂)  
  - `-1` for negative (e.g., 😡, 👎)

### 5. **Punctuation & Special Character Removal**
Simplifies tokenization and reduces dimensionality.

### 6. **Stopword Removal**
Removed common English stopwords (e.g., “the”, “is”) using NLTK.

### 7. **Stemming**
Applied **Porter Stemmer** to reduce words to their base form.  
Example: `running` → `run`

### 8. **VADER Sentiment Analysis**
Used `vaderSentiment` to extract a compound score reflecting the emotional tone of each comment (positive/negative/neutral). This was included as a separate numerical feature.

---

## 📈 Feature Engineering

To build an effective feature set for modeling, we combined:

- **TF-IDF Vectors**: Captures word importance based on frequency across documents.
- **Emoji Sentiment Scores**: Quantifies emotional signals in emoji usage.
- **VADER Sentiment Scores**: Encodes textual emotional polarity.

These features were merged into a single feature matrix for training machine learning models.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights derived from EDA:

- **Class Imbalance**: Significantly more non-toxic comments than toxic ones.
- **Word Clouds**: Toxic comments featured more offensive and aggressive words.
- **Emoji Patterns**: Negative emojis were more frequent in toxic comments.
- **VADER Distribution**: Toxic comments showed more negative sentiment.
- **Correlation Heatmap**: VADER scores moderately correlated with toxicity, validating its feature value.

---

## 🧪 Models Trained

We evaluated four popular classification models:

| Model              | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| Logistic Regression | 69.1%     | 18.1%  | 28.6%    |
| Random Forest       | 72.2%     | 12.4%  | 21.1%    |
| SVM (SVC)           | 66.0%     | 14.7%  | 24.1%    |
| **XGBoost**         | 56.5%     | **29.0%**  | **38.4%**    |

> 📌 **XGBoost** was selected for final deployment due to its superior recall and F1-score for the toxic class.

---

## 🛠️ Model Optimization

### ✅ Hyperparameter Tuning (GridSearchCV)
Best parameters selected:
- `n_estimators = 300`
- `learning_rate = 0.2`
- `max_depth = 5`
- `colsample_bytree = 1.0`
- `subsample = 0.7`

### ✅ Threshold Tuning
- Default threshold = 0.5 (missed many toxic comments)
- **Optimized threshold = 0.3** significantly improved toxic comment recall.

---

## 📊 Final Model Performance (Threshold = 0.3)

| Metric            | Class 0 (Non-toxic) | Class 1 (Toxic) |
|-------------------|---------------------|-----------------|
| Precision         | 85%                 | 40%             |
| Recall            | 62%                 | **70%**         |
| F1-Score          | 72%                 | **51%**         |
| Accuracy          | 64%                 | -               |

📌 **Confusion Matrix:**

---

## ✅ Conclusion

This project demonstrates a successful application of NLP, sentiment analysis, and machine learning to identify toxic comments online. By combining **TF-IDF**, **emoji sentiment**, and **VADER scores**, and optimizing using **XGBoost with threshold tuning**, we created a balanced solution that improves toxic comment detection with acceptable trade-offs. Future extensions may include deep learning or BERT-based models for improved contextual understanding.

---

## 📁 Project Files Description

| File Name                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `toxic_comment_classifier.ipynb`  | Main Jupyter Notebook containing the entire pipeline — data preprocessing, EDA, feature engineering (TF-IDF, VADER, emoji scores), model building (Logistic Regression, SVM, Random Forest, XGBoost), hyperparameter tuning, threshold adjustment, and prediction. |
| `project_report.pdf`              | A comprehensive report outlining the problem, methodology, models used, evaluation metrics, EDA insights, and final results. |
| `training.json`                   | Raw training dataset with text comments, platform metadata, and composite toxicity labels used for supervised learning. |
| `test.json`                       | Unlabeled test dataset used to evaluate the trained model and generate final predictions for submission. |
| `predictions.csv`                 | Final output CSV containing `platform_id` and predicted toxicity (`True` for toxic, `False` for non-toxic) on test set. |


## 📚 References

- [VADER Sentiment Analysis - GfG](https://www.geeksforgeeks.org/sentiment-analysis-of-youtube-comments/)
- [TF-IDF and NLP Basics](https://www.analyticsvidhya.com/blog/2021/06/twitter-sentiment-analysis-a-nlp-use-case-for-beginners/)
- [Project Inspiration - GitHub](https://github.com/PatilMrudu/Youtube-Comments-Sentimental-Analysis)

---
