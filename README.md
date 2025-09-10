# Financial Sentiment Analysis

This repository contains my implementation of a financial sentiment classification task using natural language processing (NLP) and machine learning. The project predicts whether a financial sentence expresses positive, negative, or neutral sentiment.

---

## Objective
To apply NLP techniques and machine learning algorithms on financial text data (FinancialPhraseBank dataset) in order to build a reliable sentiment classifier.

---

## My Experience in This Project
- **Data Handling**:  
  Imported and explored the FinancialPhraseBank dataset, inspected class distribution, and prepared text/label pairs for modeling.
  
- **Text Preprocessing**:  
  - Tokenization  
  - Lowercasing  
  - Stopword removal  
  - Lemmatization (NLTK/WordNet)  
  These steps ensured cleaner input for feature extraction.

- **Feature Extraction**:  
  Used **TF-IDF vectorization** to convert text into numerical features suitable for machine learning.

- **Model Training**:  
  Implemented **Logistic Regression** as the primary classifier for sentiment prediction.

- **Evaluation**:  
  Generated a classification report with **accuracy, precision, recall, F1-score** to measure performance.

- **Tools**:  
  Worked in Jupyter Notebook, structured workflow step by step, and documented results.

---

## Tools & Libraries Used
- **Python**  
- **Pandas, NumPy** – data manipulation  
- **NLTK** – text preprocessing, stopwords, lemmatization  
- **scikit-learn** – TF-IDF, Logistic Regression, train/test split, evaluation metrics  
- **Jupyter Notebook** – interactive development

---

## Dataset
- **Source**: [FinancialPhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)  
- **Labels**: Positive, Negative, Neutral financial sentences  
- **Usage**: Preprocessed text → TF-IDF → Logistic Regression model

---

## How to Run
```bash
git clone https://github.com/avann17/Financial-Sentiment-Analysis.git
cd Financial-Sentiment-Analysis
pip install -r requirements.txt   # or install libraries manually
jupyter notebook Financial_Sentiment_Analysis.ipynb
