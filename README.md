# Financial Sentiment Analysis

This project implements a financial sentiment analysis pipeline that processes financial texts, extracts sentiment signals, and evaluates model performance. Demonstrates skills in NLP, machine learning, and domain-specific preprocessing.

---

## Objective  
Classify financial documents or statements as positive, negative, or neutral using natural language processing techniques optimized for financial language.

---

## My Experience in This Project  
- **Text Preprocessing & Cleaning**  
  Normalized financial text, handled punctuation, tokenized effectively. Applied domain-specific adjustments (e.g., handling ticker symbols, punctuation-heavy formats).

- **Feature Engineering & Representation**  
  Represented text using TF-IDF, word embeddings, or transformer-based encodings (e.g., BERT-based models like FinBERT).

- **Model Development**  
  Explored sentiment classification via:
  - Lexicon-based approaches (e.g., VADER, custom financial lexicons)
  - Classical ML models (SVM, Random Forest, Logistic Regression)
  - Transformer-based models (FinBERT, optionally fine-tuned)

- **Model Evaluation & Analysis**  
  Evaluated using accuracy, precision, recall, F1-score. Conducted error analysis to identify misclassification patterns in financial contexts.

- **Tools & Workflow  
  Used Jupyter Notebook for experimentation and documentation. Saved trained models for reuse with `joblib` or equivalent.

---

## Tools & Libraries  
- **Python**  
- **Pandas**, **NumPy** – data manipulation  
- **NLTK**, **spaCy**, **scikit-learn** – preprocessing, classical modeling  
- **Transformers** (Hugging Face) – pre-trained transformer models (e.g., FinBERT)  
- **Joblib** or **pickle** – model persistence  
- **Jupyter Notebook** – interactive development and documentation  

---

## Data  
Dataset derived from Financial PhraseBank, FiQA, or similar financial sentiment corpora. Texts labeled as positive, negative, or neutral for training and evaluation.

---

## How to Run  
```bash
git clone https://github.com/avann17/Financial-Sentiment-Analysis.git
cd Financial-Sentiment-Analysis
pip install -r requirements.txt   # or install the listed libraries
jupyter notebook                  # open and run analysis notebook
