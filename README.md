# 🎬 Sentiment Analysis of Movie Reviews

> Machine Learning project for classifying movie reviews as **positive** or **negative**

---

## 📌 Overview
This project implements a complete **sentiment analysis pipeline** using classical machine learning methods.

It covers:
- 🧹 data cleaning & preprocessing  
- 🧠 feature extraction (TF-IDF)  
- 🤖 training multiple ML models  
- 📊 evaluation & visualization  
- 🌐 simple web app (Flask)

---

## 📊 Dataset
The model is trained on the **IMDB Dataset**.

Main preprocessing steps:
- removing duplicates  
- cleaning text (HTML, symbols, spacing)  
- removing empty reviews  
- mapping labels:
  - `positive → 1`
  - `negative → 0`

---

## ⚙️ Preprocessing

### 🔹 Basic cleaning
- remove HTML tags  
- normalize punctuation  
- remove unwanted characters  
- lowercase text  

### 🔹 Advanced processing
- tokenization  
- lemmatization (NLTK)  

---

## 🧠 Feature Engineering

We use **TF-IDF Vectorization**:
- n-grams: `(1,2)`  
- max features: `120,000`  
- filtering rare/common words  

---

## 🤖 Models

The following models are trained and compared:

- Logistic Regression  
- Linear SVM (LinearSVC)  
- Ridge Classifier  
- Multinomial Naive Bayes  

---

## 📈 Evaluation

Models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- **F1-score (main metric)**  
- ROC AUC  

Additional analysis:
- 📉 confusion matrices  
- 📈 ROC curves  
- 📊 learning curves  
- ❌ false positive / false negative analysis  
- 📊 McNemar statistical test  

---

## 🏆 Model Selection
The best model is selected based on:
- highest **F1-score**  
- generalization (overfitting penalty)  

Final model + vectorizer are saved for reuse.

---

## 🌐 Web App

A simple Flask app allows real-time predictions:

**Input:** movie review  
**Output:** sentiment (Positive / Negative)

## 📂 Project Structure
```bash
project/
│
├── data/                  # dataset
├── models/                # saved model + vectorizer
├── charts/                # plots
├── analysis/              # error analysis
├── results/               # metrics (JSON)
│
├── main.py
├── main_train.py
├── clean_data.py
├── preprocessing.py
├── training_and_calculating_metrics.py
├── analysis.py
├── app.py
├── save_to_json.py
├── requirements.txt
```

## 🚀 How to Run
1. - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r requirements.txt
3. Train model - python main.py
4. Run web app - python app.py

## 📦 Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- flask
- scipy

## 🔮 Future Improvements
- Deep learning (LSTM / Transformers)
- Hyperparameter tuning
- API deployment



