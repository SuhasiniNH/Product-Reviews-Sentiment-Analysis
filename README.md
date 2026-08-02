# Amazon Product Reviews Sentiment Analysis

Predicts whether an Amazon product review expresses positive or negative sentiment, using natural language processing and a Logistic Regression classifier trained on TF-IDF vectorized review text.

## Dataset

The dataset consists of Amazon product reviews labeled with sentiment (positive/negative). Neutral reviews were excluded to keep the classification task binary.

## Project Structure

- `notebook.ipynb` — data cleaning, text preprocessing, vectorization, model training, and evaluation.
- `app.py` — Streamlit app for an interactive live demo.
- `sentiment_model.pkl` — trained Logistic Regression model.
- `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer used to transform review text.
- `requirements.txt` — Python dependencies.

## Text Preprocessing Pipeline

1. **Cleaning** — removed HTML tags and non-alphanumeric characters.
2. **Stopword removal** — filtered out common English stopwords using NLTK.
3. **Stemming** — reduced words to their root form using the Porter Stemmer.
4. **Vectorization** — converted cleaned text into numeric features using `TfidfVectorizer` (max 5,000 features).

## Model

A `LogisticRegression` classifier (scikit-learn) trained on TF-IDF vectorized review text.

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 80.82% |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative | 0.81 | 0.88 | 0.85 | 3718 |
| Positive | 0.80 | 0.70 | 0.75 | 2532 |
| **Macro avg** | 0.81 | 0.79 | 0.80 | 6250 |
| **Weighted avg** | 0.81 | 0.81 | 0.81 | 6250 |

Limitations:
The model struggles with reviews containing positive-sounding words used in a negative context (e.g., "ruining the nice appearance"), since TF-IDF + Logistic Regression evaluates word frequency independently, without understanding negation, sentence structure, or context. For example, [this review excerpt] was predicted as 73% positive despite being a negative review, illustrating this limitation. A context-aware model (e.g., fine-tuned BERT) would likely handle such cases more accurately.

## Running the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a product review in the text box to see the model's sentiment prediction.
